use crate::engine::bank_gate::BankGates;
use crate::engine::{with_retry, with_retry_config, Engine};
use crate::entity::{
    collapse_ws, normalize_name, score_candidate, string_similarity, ResolverConfig,
};
use crate::prompts::prompt_utils::{iso_date_utc, iso_datetime_utc};
use crate::prompts::reflect::{
    build_opinion_extraction_prompt, build_reflect_system, parse_opinions, reflect_tools,
};
use crate::prompts::upstream::reflect::{
    build_chunk_claims_prompt, build_done_request_prompt, build_final_prompt,
    build_final_system_prompt, build_reduce_prompt, split_context_history, ContextHistoryEntry,
    Directive, DispositionTraits, ReflectBankProfile, CLAIMS_SYSTEM_PROMPT,
};
use crate::traits::{
    ChatMessage, Clock, EmbedProvider, LlmProvider, LlmRequest, LlmResponse, Role, ToolCall,
    ToolDef,
};
use crate::types::*;
use crate::{Error, MemoryStore, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub(super) async fn reflect(
    e: &Engine,
    bank_id: &str,
    query: &str,
    opts: &ReflectOptions,
) -> Result<ReflectResponse> {
    // Upstream's deprecated `context` field is concatenated onto the query.
    let query_owned = match opts.context.as_deref().map(str::trim) {
        Some(ctx) if !ctx.is_empty() => format!("{query}\n\nAdditional context: {ctx}"),
        _ => query.to_string(),
    };
    let query = query_owned.as_str();
    if query.trim().is_empty() {
        return Err(Error::BadInput("query is empty".into()));
    }
    let max_rounds = opts
        .max_rounds
        .unwrap_or(e.config.reflect_max_rounds)
        .max(1);
    let answer_max_tokens = opts.max_tokens.or(e.config.reflect_answer_max_tokens);
    // `fact_types` restriction: a tool whose kinds are all excluded still
    // answers, with an empty result, so the loop's contract is unchanged.
    let allowed = |kinds: &[MemoryKind]| -> Vec<MemoryKind> {
        match &opts.kinds {
            Some(only) => kinds.iter().copied().filter(|k| only.contains(k)).collect(),
            None => kinds.to_vec(),
        }
    };
    let raw_kinds = allowed(&[MemoryKind::World, MemoryKind::Experience]);
    let observation_kinds = allowed(&[MemoryKind::Observation]);
    let bank = e.store.get_bank(bank_id)?.unwrap_or(Bank {
        id: bank_id.to_string(),
        mission: None,
        directives: None,
        background: None,
        disposition: Disposition::default(),
    });
    let system = build_reflect_system(&bank);
    let mut messages: Vec<ChatMessage> = vec![ChatMessage::user(query.to_string())];
    let mut retrieved: HashSet<String> = HashSet::new();
    // Everything the tools returned, in upstream's synthesis-prompt shape —
    // the input to the final-synthesis call.
    let mut context_history: Vec<ContextHistoryEntry> = vec![];
    // The loop model's volunteered answer (finish call or plain content),
    // kept as the fallback: upstream's dominant answer path is the CLOSING
    // ASK (`_ask_for_done`) — a fresh user turn (`build_done_request_prompt`)
    // that asks for the answer over the full tool conversation — because
    // models volunteer terse literal finishes far more often than they
    // write exhaustive ones (upstream measured ~29% volunteer done; the
    // 2026-11-12 parity run: gpt-oss one-liners on point questions with
    // complete evidence in context). The closing ask always runs; the
    // standalone synthesis only runs when neither produced anything.
    let mut volunteered: Option<(String, Vec<String>)> = None;
    let mut rounds = 0usize;
    // Consecutive responses with neither content nor tool calls. One
    // nudge is allowed per streak; a second consecutive empty response
    // hands the question to the final-synthesis call over whatever was
    // gathered — it no longer fails the reflect.
    let mut empty_streak = 0usize;
    // Upstream #4567 sufficiency gate: a `finish` reporting "nothing is
    // known" (`answer_found: false`) is refused once while `recall` has
    // not run on the question; the retried finish is always honored.
    let mut recall_ran = false;
    let mut finish_gate_refused = false;
    let llm = e.llm_for_reflect();
    let loop_started = std::time::Instant::now();

    while rounds < max_rounds {
        rounds += 1;
        let request = LlmRequest {
            system: Some(system.clone()),
            messages: messages.clone(),
            tools: reflect_tools(),
            temperature: Some(0.2),
            ..Default::default()
        };
        let resp = with_retry(e, || llm.complete(&request)).await?;

        if !resp.tool_calls.is_empty() {
            empty_streak = 0;
            // Echo the whole batched turn back, then answer each call
            // with its own tool result — models that emit `recall` +
            // `finish` together (gpt-oss does this routinely) complete
            // in one round instead of having the `finish` dropped on the
            // floor by a first-call-only handler.
            messages.push(ChatMessage {
                role: Role::Assistant,
                content: String::new(),
                tool_call_id: None,
                tool_calls: resp.tool_calls.clone(),
            });
            let mut finished = false;
            for call in &resp.tool_calls {
                match call.name.as_str() {
                    "recall" => {
                        recall_ran = true;
                        let (text, entry) = run_recall_tool(
                            e,
                            bank_id,
                            &call.arguments,
                            query,
                            RecallTool {
                                kinds: &raw_kinds,
                                entry_key: "memories",
                                default_budget: DEFAULT_RECALL_BUDGET,
                                tags: opts.tags.as_ref(),
                            },
                        )
                        .await?;
                        collect(&mut retrieved, &entry, "memories");
                        context_history.push(entry);
                        messages.push(ChatMessage::tool_result(call.id.clone(), text));
                    }
                    "search_observations" => {
                        let (text, entry) = run_recall_tool(
                            e,
                            bank_id,
                            &call.arguments,
                            query,
                            RecallTool {
                                kinds: &observation_kinds,
                                entry_key: "observations",
                                default_budget: DEFAULT_SEARCH_OBSERVATIONS_BUDGET,
                                tags: opts.tags.as_ref(),
                            },
                        )
                        .await?;
                        collect(&mut retrieved, &entry, "observations");
                        context_history.push(entry);
                        messages.push(ChatMessage::tool_result(call.id.clone(), text));
                    }
                    "read_observation" => {
                        let name = call
                            .arguments
                            .get("entity")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let target = normalize_name(&name);
                        let ents = e.store.entities_for_bank(bank_id)?;
                        let (text, entry) = match ents
                            .iter()
                            .find(|en| normalize_name(&en.name) == target)
                        {
                            Some(en) => {
                                let obs =
                                    e.store.observations_for_entity(&en.id).unwrap_or_default();
                                // Only observations actually surfaced to the model
                                // are citable — never the whole bank.
                                for o in &obs {
                                    retrieved.insert(o.id.clone());
                                }
                                let entry = ContextHistoryEntry::new(
                                    "read_observation",
                                    serde_json::json!({
                                        "observations": obs.iter().map(observation_json).collect::<Vec<_>>()
                                    }),
                                );
                                let text = if obs.is_empty() {
                                    format!("(no observations for {})", en.name)
                                } else {
                                    obs.iter()
                                        .map(|o| {
                                            format!(
                                                "[observation] {}: {} (evidence: {})",
                                                o.id, o.text, o.proof_count
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                };
                                (text, entry)
                            }
                            None => (
                                format!("(unknown entity {name})"),
                                ContextHistoryEntry::new(
                                    "read_observation",
                                    serde_json::json!({"observations": []}),
                                ),
                            ),
                        };
                        context_history.push(entry);
                        messages.push(ChatMessage::tool_result(call.id.clone(), text));
                    }
                    "finish" => {
                        let answer_found =
                            call.arguments.get("answer_found").and_then(|v| v.as_bool());
                        if answer_found == Some(false) && !recall_ran && !finish_gate_refused {
                            finish_gate_refused = true;
                            messages.push(ChatMessage::tool_result(
                                call.id.clone(),
                                "finish refused: answer_found is false but `recall` has not \
                                 run on this question yet. Call `recall` with the question's \
                                 key terms before reporting that nothing is known, then call \
                                 `finish` again."
                                    .to_string(),
                            ));
                            continue;
                        }
                        loop_answer_to(
                            &mut volunteered,
                            call.arguments
                                .get("answer")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            call.arguments
                                .get("citations")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                        .collect::<Vec<String>>()
                                })
                                .unwrap_or_default(),
                            &retrieved,
                        );
                        messages.push(ChatMessage::tool_result(
                            call.id.clone(),
                            "answer recorded".to_string(),
                        ));
                        finished = true;
                    }
                    other => {
                        messages.push(ChatMessage::tool_result(
                            call.id.clone(),
                            format!("unknown tool {other}"),
                        ));
                    }
                }
                if finished {
                    break;
                }
            }
            if finished {
                break;
            }
        } else if let Some(c) = resp.content.filter(|c| !c.trim().is_empty()) {
            if looks_like_harness_noise(&c) {
                // Gateway/harness artifacts (gpt-oss channel tokens like
                // `<|channel|>final <|constrain|>citations":...]}`) are not
                // an answer — the 2026-11-12 parity run returned one
                // verbatim and the judge scored an abstain mismatch. Treat
                // the turn as empty: nudge once, then the closing ask.
                empty_streak += 1;
                if empty_streak > 1 {
                    break;
                }
                messages.push(ChatMessage::user(
                    "Your previous reply came back malformed. Call one of the \
                     provided tools (search_observations, recall, \
                     read_observation, finish) or reply with your final answer \
                     as plain text.",
                ));
                continue;
            }
            // Some models answer the `finish` tool's arguments as plain JSON
            // content instead of a tool call (seen from gpt-oss via OpenRouter
            // in the 2026-11-11 parity run). Unwrap a finish-shaped body so
            // the JSON envelope never becomes the visible answer and its
            // citations still validate against what the tools returned.
            match parse_finish_shaped_content(&c) {
                Some((text, cited, answer_found)) => {
                    if answer_found == Some(false) && !recall_ran && !finish_gate_refused {
                        finish_gate_refused = true;
                        messages.push(ChatMessage::user(
                            "Your finish was refused: answer_found is false but `recall` has \
                             not run on this question yet. Call `recall` with the question's \
                             key terms before reporting that nothing is known, then call \
                             `finish` again.",
                        ));
                        continue;
                    }
                    loop_answer_to(&mut volunteered, text, cited, &retrieved);
                }
                None => loop_answer_to(&mut volunteered, c, vec![], &retrieved),
            }
            break;
        } else {
            // Reasoning models behind gateways that strip reasoning
            // occasionally return a message with neither content nor
            // tool calls (seen from gpt-oss-20b in the fullstack smoke).
            // Nudge once; a second consecutive empty response falls
            // through to the final-synthesis call over what was gathered.
            empty_streak += 1;
            if empty_streak > 1 {
                break;
            }
            messages.push(ChatMessage::user(
                "Your previous reply came back empty. Call one of the provided tools \
                 (search_observations, recall, read_observation, finish) or reply with your \
                 final answer as plain text.",
            ));
        }
    }

    // Closing ask (upstream `_ask_for_done` + `build_done_request_prompt`):
    // ALWAYS runs when the loop stops, whether it volunteered a finish,
    // exhausted rounds, or went silent. It rides the conversation the
    // provider already holds and puts the answer-depth directive ("This is
    // the ANSWER, not a summary of it: carry over every relevant fact,
    // date and number…") in the freshest possible position — the closing
    // user turn — which is where gold's exhaustive answers come from. The
    // `finish` tool is forced (upstream `tool_choice=done`), and the
    // sufficiency gate does not apply: this is the last word. A volunteered
    // answer survives as the fallback when the ask itself fails.
    let mut closing: Option<(String, Vec<String>)> = None;
    {
        rounds += 1;
        messages.push(ChatMessage::user(build_done_request_prompt(
            query,
            answer_max_tokens,
            None,
        )));
        let request = LlmRequest {
            system: Some(system.clone()),
            messages: messages.clone(),
            tools: vec![finish_tool_def()],
            tool_choice: Some("finish".into()),
            temperature: Some(0.2),
            max_tokens: e.config.reflect_max_completion_tokens,
            ..Default::default()
        };
        if let Ok(resp) = with_retry(e, || llm.complete(&request)).await {
            if let Some(call) = resp.tool_calls.iter().find(|c| c.name == "finish") {
                loop_answer_to(
                    &mut closing,
                    call.arguments
                        .get("answer")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    call.arguments
                        .get("citations")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect::<Vec<String>>()
                        })
                        .unwrap_or_default(),
                    &retrieved,
                );
                messages.push(ChatMessage::tool_result(
                    call.id.clone(),
                    "answer recorded".to_string(),
                ));
            } else if let Some(c) = resp
                .content
                .filter(|c| !c.trim().is_empty() && !looks_like_harness_noise(c))
            {
                // Only a finish-shaped body counts (models that emit the
                // tool arguments as content); plain prose in answer to the
                // closing ask is dropped — upstream falls to the standalone
                // synthesis there, we fall to the volunteered answer first.
                if let Some((text, cited, _)) = parse_finish_shaped_content(&c) {
                    loop_answer_to(&mut closing, text, cited, &retrieved);
                }
            }
        }
    }
    let loop_ms = loop_started.elapsed().as_millis() as u64;

    // Answer precedence: closing ask → volunteered finish → standalone
    // synthesis (upstream's fallback, `build_final_prompt`). Candidates
    // that sanitize to nothing (an answer that was nothing but pasted
    // memory ids) fall through to the next path instead of failing the
    // reflect; a failure of all three means no answer exists to give.
    let mut answer: Option<(String, Vec<String>, AnswerPath)> = None;
    for (text, cited, path) in [
        closing.map(|(t, c)| (t, c, AnswerPath::ForcedDone)),
        volunteered.map(|(t, c)| (t, c, AnswerPath::Loop)),
    ]
    .into_iter()
    .flatten()
    {
        // Answer hygiene (upstream `presenter.drop_unseen_ids` + id
        // harvest): models paste bare memory ids / "Citations" sections
        // into answer text (2026-11-12 parity run). Strip id-shaped
        // tokens, drop id-only lines, harvest the stripped ids.
        let (clean, harvested) = strip_memory_ids(&text);
        if clean.trim().is_empty() {
            continue;
        }
        let mut cited_all = cited;
        cited_all.extend(harvested);
        answer = Some((clean, cited_all, path));
        break;
    }

    let (answer, citations, answer_path, synthesis_ms) = match answer {
        Some((a, c, path)) => (a, c, path, None),
        None => {
            let synthesis_started = std::time::Instant::now();
            let synthesis =
                synthesize_answer(e, &bank, query, &context_history, answer_max_tokens).await;
            let ms = synthesis_started.elapsed().as_millis() as u64;
            match synthesis {
                Ok(text) if !text.trim().is_empty() => {
                    let (clean, harvested) = strip_memory_ids(&text);
                    if clean.trim().is_empty() {
                        return Err(Error::Llm("reflect produced no answer".into()));
                    }
                    (clean, harvested, AnswerPath::Synthesis, Some(ms))
                }
                Err(err) => return Err(err),
                Ok(_) => return Err(Error::Llm("reflect produced no answer".into())),
            }
        }
    };
    let citations = validated_citations(citations, &retrieved);

    // Structured output (upstream `response_schema`): one extra JSON-mode
    // call renders the finished answer against the caller's schema. A
    // failure never fails the reflect — it is reported next to the answer.
    let (structured_output, structured_output_error) = match &opts.response_schema {
        Some(schema) => match structure_answer(e, query, &answer, schema).await {
            Ok(value) => (Some(value), None),
            Err(err) => (None, Some(err.to_string())),
        },
        None => (None, None),
    };

    // Opinion formation is best-effort housekeeping (one extraction call
    // that formed zero opinions on 29/29 parity reflects while adding a
    // full LLM round trip to each): run it in the background so the
    // response never waits on it. Errors are logged, never fatal.
    spawn_form_opinions(e, bank_id, query, &answer);

    let retrieved_count = retrieved.len();
    let mut retrieved_ids: Vec<String> = retrieved.into_iter().collect();
    retrieved_ids.sort();

    Ok(ReflectResponse {
        answer,
        citations,
        rounds,
        retrieved: retrieved_count,
        retrieved_ids,
        answer_path,
        synthesized: answer_path == AnswerPath::Synthesis,
        loop_ms,
        synthesis_ms,
        structured_output,
        structured_output_error,
    })
}

/// Render a finished reflect answer as JSON matching `schema`.
async fn structure_answer(
    e: &Engine,
    query: &str,
    answer: &str,
    schema: &serde_json::Value,
) -> Result<serde_json::Value> {
    let llm = e.llm_for_reflect();
    let request = LlmRequest {
        system: Some(
            "You convert an answer into structured data. Respond with a single JSON object \
             that conforms to the JSON Schema you are given. Use only information present in \
             the answer; use null (or omit optional fields) for anything it does not state. \
             Output JSON only."
                .to_string(),
        ),
        messages: vec![ChatMessage::user(format!(
            "JSON Schema:\n{schema}\n\nQuestion:\n{query}\n\nAnswer:\n{answer}"
        ))],
        json_mode: true,
        temperature: Some(0.0),
        max_tokens: e.config.reflect_max_completion_tokens,
        ..Default::default()
    };
    let resp = with_retry(e, || llm.complete(&request)).await?;
    let content = resp.content.unwrap_or_default();
    let raw = content.trim();
    // Tolerate a fenced block around the object.
    let raw = raw
        .strip_prefix("```json")
        .or_else(|| raw.strip_prefix("```"))
        .map(|r| r.trim_end_matches("```").trim())
        .unwrap_or(raw);
    serde_json::from_str::<serde_json::Value>(raw)
        .map_err(|err| Error::Parse(format!("structured output was not valid JSON: {err}")))
}

/// Record a loop-produced answer: keep the text and its validated
/// citations together so the winning path's citations are the ones
/// returned.
fn loop_answer_to(
    slot: &mut Option<(String, Vec<String>)>,
    text: String,
    cited: Vec<String>,
    retrieved: &HashSet<String>,
) {
    *slot = Some((text, validated_citations(cited, retrieved)));
}

/// Harness/gateway artifacts that must never become an answer: gpt-oss
/// special tokens leaking through OpenRouter as content (the 2026-11-12
/// run returned `<|channel|>final <|constrain|>citations":…]}` verbatim
/// for one question). Upstream normalizes the same tokens on tool names
/// (`_normalize_tool_name`'s `done<|channel|>commentary` case).
fn looks_like_harness_noise(text: &str) -> bool {
    [
        "<|start|>",
        "<|channel|>",
        "<|call|>",
        "<|constrain|>",
        "<|end|>",
        "<|system|>",
    ]
    .iter()
    .any(|m| text.contains(m))
}

/// Remove memory-id-shaped tokens from answer text (upstream's
/// `presenter.drop_unseen_ids`): a whitespace-delimited token whose only
/// non-wrapper content is a UUID (optionally wrapped in markdown/punctuation)
/// is dropped; lines left holding nothing but a citations label go too.
/// Returns the cleaned text and the harvested ids.
fn strip_memory_ids(text: &str) -> (String, Vec<String>) {
    let is_hex = |c: char| c.is_ascii_hexdigit();
    let shape_uuid = |t: &str| -> Option<String> {
        let core: String = t
            .chars()
            .filter(|&c| c.is_ascii_hexdigit() || c == '-')
            .collect();
        let parts: Vec<&str> = core.split('-').collect();
        if parts.len() == 5
            && parts[0].len() == 8
            && parts[1].len() == 4
            && parts[2].len() == 4
            && parts[3].len() == 4
            && parts[4].len() == 12
            && core.chars().all(|c| c == '-' || is_hex(c))
        {
            Some(core.to_lowercase())
        } else {
            None
        }
    };
    let citation_label = |line: &str| -> bool {
        let bare = line
            .trim()
            .trim_matches(|c: char| {
                c == '*' || c == '#' || c == '_' || c == ':' || c == '-' || c.is_whitespace()
            })
            .to_lowercase();
        matches!(
            bare.as_str(),
            "citations" | "citation" | "sources" | "references" | "memory ids" | "ids" | "evidence"
        )
    };

    let mut harvested: Vec<String> = vec![];
    let mut out_lines: Vec<String> = vec![];
    for line in text.lines() {
        let mut kept: Vec<String> = vec![];
        for token in line.split_whitespace() {
            match shape_uuid(token) {
                Some(id) => harvested.push(id),
                None => kept.push(token.to_string()),
            }
        }
        let joined = kept.join(" ");
        let decoration_only = joined
            .trim()
            .trim_matches(|c: char| {
                c == '-' || c == '*' || c == '.' || c == '•' || c == ':' || c.is_whitespace()
            })
            .is_empty();
        if decoration_only {
            // A line that held only ids and bullet/label decoration (or was
            // already blank): keep blanks, drop the id-only ones entirely.
            if line.trim().is_empty() {
                out_lines.push(String::new());
            }
            continue;
        }
        if citation_label(&joined) {
            // A "Citations:"-style header (or one left behind by id
            // removal) — answers carry ids only in the citations array.
            continue;
        }
        out_lines.push(joined);
    }
    let mut cleaned = out_lines.join("\n");
    // Collapse runs of blank lines left by removals.
    while cleaned.contains("\n\n\n") {
        cleaned = cleaned.replace("\n\n\n", "\n\n");
    }
    cleaned = cleaned.trim().to_string();
    (cleaned, harvested)
}

/// Tool token-budget defaults, mirroring upstream's tool schemas: the
/// observation search returns up to 5000 result tokens, the raw-facts
/// recall 2048 (both overridable per call by the model).
const DEFAULT_SEARCH_OBSERVATIONS_BUDGET: u64 = 5000;
const DEFAULT_RECALL_BUDGET: u64 = 2048;

fn finish_tool_def() -> ToolDef {
    ToolDef {
        name: "finish".into(),
        description: "Deliver the final answer. Cite only memory ids that were returned by \
                      tools."
            .into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your final answer"},
                "citations": {"type": "array", "items": {"type": "string"}, "description": "Memory ids actually used"},
                "answer_found": {"type": "boolean", "description": "true if what you retrieved states the answer to the question. false if you are reporting that nothing (or not enough) is known about it."}
            },
            "required": ["answer"]
        }),
    }
}

/// Run one `recall`-shaped tool (`recall` for raw facts,
/// `search_observations` for consolidated knowledge): the engine recall
/// restricted to the tool's kinds, rendered both as the loop-conversation
/// text and as a synthesis-prompt context entry.
/// What distinguishes the loop's two retrieval tools.
struct RecallTool<'a> {
    kinds: &'a [MemoryKind],
    /// Key the results are filed under in the context history.
    entry_key: &'a str,
    /// Result token budget when the model names none.
    default_budget: u64,
    tags: Option<&'a TagFilter>,
}

async fn run_recall_tool(
    e: &Engine,
    bank_id: &str,
    args: &serde_json::Value,
    fallback_query: &str,
    tool: RecallTool<'_>,
) -> Result<(String, ContextHistoryEntry)> {
    let RecallTool {
        kinds,
        entry_key,
        default_budget,
        tags,
    } = tool;
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_query)
        .to_string();
    let budget = args
        .get("token_budget")
        .and_then(|v| v.as_u64())
        .unwrap_or(default_budget) as usize;
    // Pool width comes from the recall budget config (upstream
    // `_resolve_thinking_budget`: Budget None → MID; fixed function
    // default = 300 per arm).
    let top_k = e.config.recall_budget.resolve(None, budget.min(6000));
    let r = if kinds.is_empty() {
        // Every kind this tool serves was excluded by `fact_types`.
        RecallResponse {
            memories: vec![],
            window: None,
            strategies_used: vec![],
        }
    } else {
        e.recall(RecallRequest {
            bank_id: bank_id.to_string(),
            query,
            token_budget: budget.min(6000),
            top_k_per_arm: top_k,
            kinds: Some(kinds.to_vec()),
            tags: tags.cloned(),
            now_ms: None,
        })
        .await?
    };
    let mut text = String::new();
    for m in &r.memories {
        let when = m
            .mentioned_at_ms
            .map(|ms| format!(" (mentioned {})", iso_date_utc(ms)))
            .unwrap_or_default();
        text.push_str(&format!(
            "[{}] {}{}: {}\n",
            m.kind.as_str(),
            m.id,
            when,
            m.text
        ));
    }
    if text.is_empty() {
        text = "(no memories found)".to_string();
    }
    let entry = ContextHistoryEntry::new(
        if entry_key == "observations" {
            "search_observations"
        } else {
            "recall"
        },
        serde_json::json!({
            entry_key: r.memories.iter().map(memory_ref_json).collect::<Vec<_>>()
        }),
    );
    Ok((text, entry))
}

fn memory_ref_json(m: &MemoryRef) -> serde_json::Value {
    serde_json::json!({
        "id": m.id,
        "kind": m.kind.as_str(),
        "text": m.text,
        "occurred_start": m.occurred_start_ms.map(iso_date_utc),
        "occurred_end": m.occurred_end_ms.map(iso_date_utc),
        "mentioned_at": m.mentioned_at_ms.map(iso_date_utc),
    })
}

fn observation_json(o: &Observation) -> serde_json::Value {
    serde_json::json!({
        "id": o.id,
        "text": o.text,
        "proof_count": o.proof_count,
        "updated_at": iso_date_utc(o.updated_at_ms),
    })
}

/// Track every id the tools surfaced (the citation-validation set).
fn collect(retrieved: &mut HashSet<String>, entry: &ContextHistoryEntry, key: &str) {
    if let Some(list) = entry.output.get(key).and_then(|v| v.as_array()) {
        for item in list {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                retrieved.insert(id.to_string());
            }
        }
    }
}

/// The final-synthesis call over the gathered tool-result history —
/// upstream's `build_final_prompt` (+ `build_final_system_prompt`). A
/// history over the context budget splits into chunks, each maps to dated
/// cited claims (`build_chunk_claims_prompt`), and one reduce call
/// (`build_reduce_prompt`) synthesizes the answer — evidence is split
/// across chunks, never dropped.
async fn synthesize_answer(
    e: &Engine,
    bank: &Bank,
    query: &str,
    history: &[ContextHistoryEntry],
    answer_max_tokens: Option<usize>,
) -> Result<String> {
    let tokens = Arc::clone(&e.tokens);
    let count = move |s: &str| tokens.count(s);
    let profile = ReflectBankProfile {
        name: Some(bank.id.as_str()),
        mission: bank.mission.as_deref(),
        disposition: Some(DispositionTraits {
            skepticism: Some(bank.disposition.skepticism as i32),
            literalism: Some(bank.disposition.literalism as i32),
            empathy: Some(bank.disposition.empathy as i32),
        }),
    };
    let directives: Vec<Directive> = bank
        .directives
        .as_deref()
        .filter(|d| !d.trim().is_empty())
        .map(|d| vec![Directive::new("", d)])
        .unwrap_or_default();
    let now_str = iso_datetime_utc(e.clock.now_ms());
    let system = build_final_system_prompt(
        bank.mission.as_deref().filter(|m| !m.trim().is_empty()),
        None,
        &directives,
        &now_str,
    );
    let llm = e.llm_for_reflect();
    let total: usize = history
        .iter()
        .map(|entry| count(&render_block(entry)))
        .sum();

    if total <= e.config.reflect_max_context_tokens {
        let user = build_final_prompt(
            query,
            history,
            &profile,
            bank.background.as_deref(),
            e.config.reflect_max_context_tokens,
            answer_max_tokens,
            None,
            &count,
        );
        let request = LlmRequest {
            system: Some(system),
            messages: vec![ChatMessage::user(user)],
            temperature: Some(0.2),
            max_tokens: e.config.reflect_max_completion_tokens,
            ..Default::default()
        };
        let resp = with_retry(e, || llm.complete(&request)).await?;
        return Ok(resp.content.unwrap_or_default());
    }

    let chunks = split_context_history(history, e.config.reflect_max_context_tokens, &count);
    let mut sections: Vec<String> = vec![];
    for chunk in &chunks {
        let user = build_chunk_claims_prompt(query, chunk);
        let request = LlmRequest {
            system: Some(CLAIMS_SYSTEM_PROMPT.to_string()),
            messages: vec![ChatMessage::user(user)],
            temperature: Some(0.0),
            max_tokens: e.config.reflect_max_completion_tokens,
            ..Default::default()
        };
        let resp = with_retry(e, || llm.complete(&request)).await?;
        sections.push(resp.content.unwrap_or_default());
    }
    let user = build_reduce_prompt(
        query,
        &sections,
        &profile,
        bank.background.as_deref(),
        answer_max_tokens,
        None,
    );
    let request = LlmRequest {
        system: Some(system),
        messages: vec![ChatMessage::user(user)],
        temperature: Some(0.2),
        max_tokens: e.config.reflect_max_completion_tokens,
        ..Default::default()
    };
    let resp = with_retry(e, || llm.complete(&request)).await?;
    Ok(resp.content.unwrap_or_default())
}

fn render_block(entry: &ContextHistoryEntry) -> String {
    crate::prompts::upstream::reflect::render_history_block(entry)
}

/// Citations are only kept when the id was actually surfaced to the model by
/// a tool in this conversation; survivors are sorted and deduped.
fn validated_citations(cited: Vec<String>, retrieved: &HashSet<String>) -> Vec<String> {
    let mut valid: Vec<String> = cited
        .into_iter()
        .filter(|id| retrieved.contains(id))
        .collect();
    valid.sort();
    valid.dedup();
    valid
}

/// Detect a `finish` tool's arguments emitted as plain content: a JSON object
/// (possibly wrapped in prose/fences) carrying an `answer` string, an
/// optional `citations` array, and an optional `answer_found` bool.
/// Anything else — genuine prose answers, other JSON shapes — returns None
/// so it passes through as the answer verbatim.
fn parse_finish_shaped_content(raw: &str) -> Option<(String, Vec<String>, Option<bool>)> {
    let body = match (raw.find('{'), raw.rfind('}')) {
        (Some(s), Some(e)) if e > s => &raw[s..=e],
        _ => return None,
    };
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    if !v.is_object() {
        return None;
    }
    let answer = v.get("answer")?.as_str()?.trim().to_string();
    if answer.is_empty() {
        return None;
    }
    let cited = v
        .get("citations")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    let answer_found = v.get("answer_found").and_then(|v| v.as_bool());
    Some((answer, cited, answer_found))
}

/// The handles the backgrounded opinion extraction needs — everything
/// `form_opinions` used off `Engine`, cloned into a detached task.
struct OpinionCtx {
    gates: Arc<BankGates>,
    store: Arc<dyn MemoryStore>,
    llm: Arc<dyn LlmProvider>,
    embed: Arc<dyn EmbedProvider>,
    clock: Arc<dyn Clock>,
    entity_cfg: ResolverConfig,
    retry_max_attempts: usize,
    retry_base_delay_ms: u64,
    max_completion_tokens: Option<u64>,
}

/// Spawn opinion formation off the response path: one extraction call (plus
/// an embed for anything it finds) that no caller should wait on — the
/// 2026-11-11 parity run saw 29/29 reflects form zero opinions while paying
/// a full LLM round trip each. Best-effort: errors are logged, never
/// surfaced to the reflect response.
fn spawn_form_opinions(e: &Engine, bank_id: &str, question: &str, answer: &str) {
    if tokio::runtime::Handle::try_current().is_err() {
        return;
    }
    let ctx = OpinionCtx {
        gates: Arc::clone(&e.bank_gates),
        store: Arc::clone(&e.store),
        llm: e.llm_for_reflect(),
        embed: Arc::clone(&e.embed),
        clock: Arc::clone(&e.clock),
        entity_cfg: e.config.entity,
        retry_max_attempts: e.config.retry_max_attempts,
        retry_base_delay_ms: e.config.retry_base_delay_ms,
        max_completion_tokens: e.config.reflect_max_completion_tokens,
    };
    let bank_id = bank_id.to_string();
    let question = question.to_string();
    let answer = answer.to_string();
    tokio::spawn(async move {
        // The answer may outlive its bank: never write opinions into one
        // that was deleted in the meantime.
        let guard = ctx.gates.read(&bank_id).await;
        if !matches!(ctx.store.get_bank(&bank_id), Ok(Some(_))) {
            return;
        }
        let formed = guard.until_deleted(form_opinions(&ctx, &bank_id, &question, &answer));
        if let Err(err) = formed.await {
            eprintln!("rustsight: opinion extraction for bank {bank_id} failed: {err}");
        }
    });
}

async fn form_opinions(
    ctx: &OpinionCtx,
    bank_id: &str,
    question: &str,
    answer: &str,
) -> Result<usize> {
    let (system, user) = build_opinion_extraction_prompt(question, answer);
    let llm = Arc::clone(&ctx.llm);
    let request = LlmRequest {
        system: Some(system),
        messages: vec![ChatMessage::user(user)],
        json_mode: true,
        temperature: Some(0.0),
        max_tokens: ctx.max_completion_tokens,
        ..Default::default()
    };
    let resp = with_retry_config(ctx.retry_max_attempts, ctx.retry_base_delay_ms, || {
        llm.complete(&request)
    })
    .await?;
    let parsed = parse_opinions(&resp.content.unwrap_or_default())?;
    if parsed.is_empty() {
        return Ok(0);
    }
    let existing = ctx.store.list_opinions(bank_id)?;
    let bank_entities = ctx.store.entities_for_bank(bank_id)?;
    let mut resolved_cache: HashMap<String, Option<Entity>> = HashMap::new();
    let now = ctx.clock.now_ms();
    let mut formed = 0usize;
    let mut new_opinions: Vec<(String, f32, Vec<String>)> = vec![];
    for op in parsed {
        let entity_ids = resolve_opinion_entities(
            ctx,
            bank_id,
            &op.entities,
            &bank_entities,
            &mut resolved_cache,
        )?;
        let similar = existing
            .iter()
            .find(|o| string_similarity(&op.text, &o.text) >= 0.85);
        match similar {
            Some(o) => {
                let mut updated = o.clone();
                updated.history.push(ConfidencePoint {
                    confidence: o.confidence,
                    at_ms: now,
                    reason: "reinforced by reflect".into(),
                });
                updated.confidence = ((o.confidence + op.confidence) / 2.0).clamp(0.0, 1.0);
                updated.updated_at_ms = now;
                for id in entity_ids {
                    if !updated.entity_ids.contains(&id) {
                        updated.entity_ids.push(id);
                    }
                }
                ctx.store.upsert_opinion(&updated)?;
            }
            None => {
                formed += 1;
                new_opinions.push((op.text, op.confidence, entity_ids));
            }
        }
    }
    if !new_opinions.is_empty() {
        let embed = Arc::clone(&ctx.embed);
        let texts: Vec<String> = new_opinions.iter().map(|(t, _, _)| t.clone()).collect();
        let embeddings = with_retry_config(ctx.retry_max_attempts, ctx.retry_base_delay_ms, || {
            embed.embed(texts.clone())
        })
        .await?;
        for ((text, confidence, entity_ids), emb) in new_opinions.into_iter().zip(embeddings) {
            let o = Opinion {
                id: uuid::Uuid::new_v4().to_string(),
                bank_id: bank_id.to_string(),
                text,
                confidence: confidence.clamp(0.0, 1.0),
                evidence: vec![],
                history: vec![],
                entity_ids,
                updated_at_ms: now,
                created_at_ms: now,
                embedding: Some(emb),
            };
            ctx.store.upsert_opinion(&o)?;
        }
    }
    Ok(formed)
}

/// Resolve LLM-tagged entity mentions in an opinion against the bank's
/// canonical entities: exact alias match first, then fuzzy match above the
/// resolver's link threshold. Unmatched mentions are dropped (never linked,
/// never created) so opinions without known entities stay unlinked and are
/// skipped by consolidation-time reinforcement.
fn resolve_opinion_entities(
    ctx: &OpinionCtx,
    bank_id: &str,
    mentions: &[ExtractedEntity],
    bank_entities: &[Entity],
    cache: &mut HashMap<String, Option<Entity>>,
) -> Result<Vec<String>> {
    let now = ctx.clock.now_ms();
    let mut ids: Vec<String> = vec![];
    for m in mentions {
        let norm = normalize_name(&m.name);
        if norm.is_empty() {
            continue;
        }
        let hit = if let Some(h) = cache.get(&norm) {
            h.clone()
        } else {
            let h = match ctx.store.find_by_alias(bank_id, &norm)? {
                Some(x) => Some(x),
                None => fuzzy_match_entity(ctx, bank_id, &norm, &m.name, bank_entities, now)?,
            };
            cache.insert(norm, h.clone());
            h
        };
        if let Some(ent) = hit {
            if !ids.contains(&ent.id) {
                ids.push(ent.id);
            }
        }
    }
    Ok(ids)
}

fn fuzzy_match_entity(
    ctx: &OpinionCtx,
    bank_id: &str,
    norm: &str,
    display: &str,
    candidates: &[Entity],
    now: i64,
) -> Result<Option<Entity>> {
    // Same scoring model as retain's resolver, minus the co-occurrence
    // signal — an opinion's entity mentions carry no same-fact siblings, so
    // the vote set is empty (exactly upstream's behavior for a mention with
    // no context: name similarity + recency only).
    let cfg: &ResolverConfig = &ctx.entity_cfg;
    let mut best: Option<(f32, &Entity)> = None;
    let empty = HashSet::new();
    let degrees = HashMap::new();
    for c in candidates {
        let Some(score) = score_candidate(
            &collapse_ws(display).to_lowercase(),
            &collapse_ws(&c.name).to_lowercase(),
            &empty,
            None,
            &degrees,
            now,
            c.last_mentioned_ms,
            cfg,
        ) else {
            continue;
        };
        if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
            best = Some((score, c));
        }
    }
    if let Some((score, ent)) = best {
        if (score * 1_000_000.0).round() / 1_000_000.0 >= cfg.link_threshold {
            ctx.store.add_alias(bank_id, norm, &ent.id)?;
            return Ok(Some(ent.clone()));
        }
    }
    Ok(None)
}

#[allow(dead_code)]
fn demo_tool_call(id: &str, name: &str, args: serde_json::Value) -> LlmResponse {
    LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: args,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn demo_tool_call_shape() {
        let r = demo_tool_call("1", "finish", json!({"answer": "hi"}));
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "finish");
    }

    #[test]
    fn finish_shaped_content_is_unwrapped() {
        let raw =
            r#"{"answer": "Alice works at Google.", "citations": ["a"], "answer_found": true}"#;
        let (answer, cited, answer_found) = parse_finish_shaped_content(raw).unwrap();
        assert_eq!(answer, "Alice works at Google.");
        assert_eq!(cited, vec!["a".to_string()]);
        assert_eq!(answer_found, Some(true));

        // Fenced / prose-wrapped envelopes too
        let fenced = format!("```json\n{raw}\n```");
        assert!(parse_finish_shaped_content(&fenced).is_some());
    }

    #[test]
    fn plain_content_is_not_finish_shaped() {
        // Prose answers, other JSON shapes, and empty answers pass through.
        assert!(parse_finish_shaped_content("Alice works at Google.").is_none());
        assert!(parse_finish_shaped_content(r#"{"opinions":[]}"#).is_none());
        assert!(parse_finish_shaped_content(r#"{"answer": "", "citations": []}"#).is_none());
        assert!(parse_finish_shaped_content(r#"{"answer": 42}"#).is_none());
    }

    #[test]
    fn citations_validate_against_retrieved() {
        let mut retrieved = HashSet::new();
        retrieved.insert("real".to_string());
        let valid = validated_citations(
            vec!["bogus".into(), "real".into(), "real".into()],
            &retrieved,
        );
        assert_eq!(valid, vec!["real".to_string()]);
    }

    #[test]
    fn harness_noise_is_detected() {
        assert!(looks_like_harness_noise(
            "<|start|>assistant<|channel|>final <|constrain|>citations\":[\"x\"]}<|call|>"
        ));
        assert!(looks_like_harness_noise("done<|channel|>commentary"));
        assert!(!looks_like_harness_noise("Alice works at Google."));
        assert!(!looks_like_harness_noise(
            r#"{"answer": "plain json is fine"}"#
        ));
    }

    #[test]
    fn inline_ids_are_stripped_and_harvested() {
        let raw = "Alice works at Google.\n\n**Citations**\n- 2aeb87d1-9b14-48d7-98cd-d822bd5c7475\n- (87579608-667e-48f2-aa36-5c922b7b7211)\n\nShe cycles to work.";
        let (clean, harvested) = strip_memory_ids(raw);
        assert_eq!(clean, "Alice works at Google.\n\nShe cycles to work.");
        assert_eq!(harvested.len(), 2);
        assert!(harvested.contains(&"2aeb87d1-9b14-48d7-98cd-d822bd5c7475".to_string()));

        // Uppercase ids normalize; id-only lines vanish with their label
        let (clean, harvested) =
            strip_memory_ids("Sources\n- 2AEB87D1-9B14-48D7-98CD-D822BD5C7475");
        assert_eq!(clean, "");
        assert_eq!(harvested, vec!["2aeb87d1-9b14-48d7-98cd-d822bd5c7475"]);
    }

    #[test]
    fn non_id_text_is_untouched() {
        let raw =
            "Joe wrote 3 books between 2020-2025. IDs like ISO-8601 stay. So do 4-digit years.";
        let (clean, harvested) = strip_memory_ids(raw);
        assert_eq!(clean, raw);
        assert!(harvested.is_empty());
    }
}
