use async_trait::async_trait;
use rustsight_core::config_resolver::Layer;
use rustsight_core::Engine;
use rustsight_core::*;
use rustsight_providers::{
    FlakyEmbed, FlakyLlm, MockEmbed, MockLlm, MockRerank, NoopRerank, SimpleTokenEstimator,
};
use rustsight_storage::SqliteStore;
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn engine() -> Engine {
    let store = SqliteStore::open_in_memory().unwrap();
    let (llm, embed, rerank) = (
        Arc::new(MockLlm::demo()) as Arc<dyn LlmProvider>,
        Arc::new(MockEmbed::default()) as Arc<dyn EmbedProvider>,
        Arc::new(MockRerank) as Arc<dyn RerankProvider>,
    );
    Engine::new(
        Arc::new(store),
        llm,
        embed,
        rerank,
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
}

async fn retain(e: &Engine, bank: &str, content: &str) -> RetainResponse {
    e.retain(RetainRequest {
        bank_id: bank.into(),
        content: content.into(),
        context: None,
        timestamp_ms: None,
        ..Default::default()
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn retain_extracts_and_recall_finds() {
    let e = engine();
    let r = retain(
        &e,
        "demo",
        "Last week Alice joined Google as a site reliability engineer. Bob enjoys hiking in Yosemite.",
    )
    .await;
    assert!(r.memories_stored >= 2, "{r:?}");
    assert!(!r.entities.is_empty());
    assert!(r.edges_created >= 1, "{r:?}");

    let resp = e
        .recall(RecallRequest {
            bank_id: "demo".into(),
            query: "Where does Alice work?".into(),
            token_budget: 500,
            top_k_per_arm: 10,
            kinds: None,
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!resp.memories.is_empty(), "{resp:?}");
    assert!(
        resp.memories.iter().any(|m| m.text.contains("Alice")),
        "alice fact missing from results: {resp:?}"
    );
    assert!(
        resp.strategies_used.iter().any(|s| s == "semantic")
            || resp.strategies_used.iter().any(|s| s == "keyword")
    );
}

#[tokio::test]
async fn entity_resolution_merges_aliases() {
    let e = engine();
    retain(&e, "b", "Alice joined Google in March.").await;
    retain(&e, "b", "Alice later got promoted at Google.").await;
    let ents = e.store.entities_for_bank("b").unwrap();
    let alices = ents
        .iter()
        .filter(|x| x.name.eq_ignore_ascii_case("alice"))
        .count();
    let googles = ents
        .iter()
        .filter(|x| x.name.eq_ignore_ascii_case("google"))
        .count();
    assert_eq!(alices, 1);
    assert_eq!(googles, 1);
}

#[tokio::test]
async fn temporal_window_recall() {
    let e = engine();
    e.retain(RetainRequest {
        bank_id: "t".into(),
        content: "CONTENT marker: Alice visited Rome.".into(),
        context: None,
        timestamp_ms: Some(1_683_648_000_000),
        ..Default::default()
    })
    .await
    .unwrap();
    let resp = e
        .recall(RecallRequest {
            bank_id: "t".into(),
            query: "What happened in 2023?".into(),
            token_budget: 500,
            top_k_per_arm: 10,
            kinds: None,
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(resp.window.is_some());
    assert!(resp.strategies_used.iter().any(|s| s == "temporal"));
    assert!(!resp.memories.is_empty(), "{resp:?}");
    assert!(
        resp.memories[0].text.contains("Rome"),
        "top memory was: {:?} (all: {:?})",
        resp.memories[0],
        resp.memories
    );
}

#[tokio::test]
async fn consolidation_builds_observation_with_evidence() {
    let e = engine();
    retain(
        &e,
        "c",
        "Alice joined Google. Alice works on search infrastructure.",
    )
    .await;
    let updated = e.consolidate_once("c", 10).await.unwrap();
    assert!(updated >= 1, "updated={updated}");
    let obs = e.store.observations_for_bank("c").unwrap();
    assert_eq!(obs.len(), 1);
    assert!(!obs[0].evidence.is_empty());
    assert!(obs[0].proof_count >= 1);
    assert!(e.store.list_dirty_entities("c", 10).unwrap().is_empty());
}

/// CR-5: the recall arms run concurrently now, but `strategies_used` must
/// keep the fixed semantic → keyword → graph → temporal order. The ghost
/// edge exists because the graph arm can only surface ids outside the
/// semantic/BM25 seed set (seeds count toward the neighbor limit), and in a
/// bank this small every real memory already sits in those seeds.
#[tokio::test]
async fn recall_strategies_used_keeps_fixed_order() {
    let e = engine();
    e.store
        .ensure_bank(&Bank {
            id: "so".into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
    let may2023 = 1_683_648_000_000i64;
    let emb = e
        .embed
        .embed(vec![
            "Alice works at Google".into(),
            "Bob likes quiet trails".into(),
        ])
        .await
        .unwrap();
    let stored = e
        .store
        .insert_memories(
            "so",
            &[
                NewMemoryRecord {
                    kind: MemoryKind::World,
                    text: "Alice works at Google".into(),
                    confidence: None,
                    occurred_start_ms: Some(may2023),
                    occurred_end_ms: None,
                    mentioned_at_ms: may2023,
                    embedding: emb[0].clone(),
                    source: None,
                    context: None,
                    document_id: None,
                    tags: vec![],
                    metadata: None,
                    chunk_id: None,
                },
                NewMemoryRecord {
                    kind: MemoryKind::World,
                    text: "Bob likes quiet trails".into(),
                    confidence: None,
                    occurred_start_ms: None,
                    occurred_end_ms: None,
                    mentioned_at_ms: may2023 + 1000,
                    embedding: emb[1].clone(),
                    source: None,
                    context: None,
                    document_id: None,
                    tags: vec![],
                    metadata: None,
                    chunk_id: None,
                },
            ],
        )
        .unwrap();
    e.store
        .add_edges(&[Edge {
            bank_id: "so".into(),
            a_id: stored[0].id.clone(),
            b_id: "graph-only-ghost".into(),
            kind: EdgeKind::Entity,
            label: None,
            weight: 1.0,
        }])
        .unwrap();

    let resp = e
        .recall(RecallRequest {
            bank_id: "so".into(),
            query: "Where did Alice work in 2023?".into(),
            token_budget: 500,
            top_k_per_arm: 3,
            kinds: None,
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        resp.strategies_used,
        vec!["semantic", "keyword", "graph", "temporal"]
    );
    assert!(resp.window.is_some());
    assert!(!resp.memories.is_empty(), "{resp:?}");
}

/// CR-6: the embedding backfill targets observations by id, so after a
/// pass every observation carries an embedding — including ones created via
/// the update path on a later pass.
#[tokio::test]
async fn consolidation_backfills_observation_embeddings_by_id() {
    let e = engine();
    retain(
        &e,
        "emb",
        "Alice joined Google. Alice works on search infrastructure.",
    )
    .await;
    e.consolidate_once("emb", 10).await.unwrap();
    let obs = e.store.observations_for_bank("emb").unwrap();
    assert!(!obs.is_empty());
    for o in &obs {
        assert!(
            o.embedding.is_some(),
            "embedding missing on observation {} ({:?})",
            o.id,
            o.text
        );
    }

    // A second retain dirties the entity again; whether the next pass
    // updates the existing observation or writes a new one, every
    // observation must still end up embedded.
    retain(
        &e,
        "emb",
        "Alice delivered the migration ahead of schedule.",
    )
    .await;
    e.consolidate_once("emb", 10).await.unwrap();
    for o in e.store.observations_for_bank("emb").unwrap() {
        assert!(
            o.embedding.is_some(),
            "embedding missing on observation {} ({:?})",
            o.id,
            o.text
        );
    }
}

#[tokio::test]
async fn background_worker_consolidates_after_retain_without_explicit_call() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        facts_resp(&[
            ("Alice joined Google as an engineer.", "Alice"),
            ("Alice works on search infrastructure.", "Alice"),
        ]),
        LlmResponse {
            content: Some(
                r#"{"observations":[{"text":"Summary: Alice works at Google on search infrastructure.","evidence":[{"fact_index":0,"quote":""}]}]}"#
                    .into(),
            ),
            tool_calls: vec![],
        },
    ]);
    let engine = Arc::new(Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    ));
    // Long interval: only the retain-time wakeup (not the periodic tick) can
    // consolidate within the deadline, so this proves the notify path.
    let worker = engine.spawn_worker(Duration::from_secs(3600), 10);
    retain(
        &engine,
        "w",
        "Alice joined Google as an engineer. Alice works on search infrastructure.",
    )
    .await;

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let obs = engine.store.observations_for_bank("w").unwrap();
        if !obs.is_empty() {
            assert_eq!(obs.len(), 1);
            assert_eq!(
                obs[0].text,
                "Summary: Alice works at Google on search infrastructure."
            );
            assert!(!obs[0].evidence.is_empty());
            break;
        }
        assert!(
            Instant::now() < deadline,
            "worker never consolidated after retain"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(engine
        .store
        .list_dirty_entities("w", 10)
        .unwrap()
        .is_empty());
    worker.abort();
    let _ = worker.await;
}

fn tool_call(id: &str, name: &str, args: serde_json::Value) -> LlmResponse {
    LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: args,
        }],
    }
}

/// One turn of the final-synthesis call (upstream's answer machine) — the
/// scripted sequence for every reflect is: loop turns → [forced-done] →
/// synthesis → opinion extraction.
fn synthesis_resp(text: &str) -> LlmResponse {
    LlmResponse {
        content: Some(text.into()),
        tool_calls: vec![],
    }
}

fn extraction_resp() -> LlmResponse {
    LlmResponse {
        content: Some(
            json!({"facts":[{"text":"Alice works at Google as an engineer.","fact_type":"world","entities":[{"name":"Alice","etype":"person"},{"name":"Google","etype":"organization"}],"occurred_start":null,"occurred_end":null,"causal":[]}]}).to_string(),
        ),
        tool_calls: vec![],
    }
}

#[tokio::test]
async fn reflect_loop_with_scripted_tools() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        extraction_resp(),
        tool_call("1", "recall", json!({"query": "Alice"})),
        tool_call(
            "2",
            "finish",
            json!({"answer": "Alice works at Google.", "citations": ["bogus-id"]}),
        ),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(&e, "r", "Alice works at Google as an engineer.").await;
    let resp = e.reflect("r", "What does Alice do?").await.unwrap();
    assert_eq!(resp.answer, "Alice works at Google.");
    // Upstream done-first precedence: the loop's own finish answer wins;
    // no standalone synthesis call runs.
    assert_eq!(resp.answer_path, AnswerPath::Loop);
    assert!(!resp.synthesized);
    assert!(resp.rounds >= 2);
    assert!(resp.retrieved >= 1);
    assert!(
        resp.citations.is_empty(),
        "bogus citation must be filtered: {resp:?}"
    );
}

#[tokio::test]
async fn reflect_unwraps_finish_shaped_content_answers() {
    // 2026-11-11 parity-run failure mode: gpt-oss emitted the `finish`
    // arguments as plain JSON content instead of a tool call, so the raw
    // `{"answer": ..., "citations": [...]}` envelope became the visible
    // answer and citations validated to zero. The loop must unwrap the
    // envelope and validate its citations like a real finish call.
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let mk_engine = |llm: Arc<dyn LlmProvider>| {
        Engine::new(
            store.clone(),
            llm,
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(1_800_000_000_000)),
        )
    };
    let seeder = mk_engine(Arc::new(MockLlm::script(vec![extraction_resp()])));
    retain(&seeder, "fj", "Alice works at Google as an engineer.").await;
    let real_id = seeder
        .recall(RecallRequest {
            bank_id: "fj".into(),
            query: "Alice".into(),
            token_budget: 500,
            top_k_per_arm: 5,
            kinds: None,
            ..Default::default()
        })
        .await
        .unwrap()
        .memories[0]
        .id
        .clone();

    let script = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "Alice"})),
        LlmResponse {
            content: Some(format!(
                r#"{{"answer": "Alice works at Google.", "citations": ["{real_id}", "bogus-id"]}}"#
            )),
            tool_calls: vec![],
        },
    ]);
    let e = mk_engine(Arc::new(script));
    let resp = e.reflect("fj", "What does Alice do?").await.unwrap();
    assert_eq!(
        resp.answer, "Alice works at Google.",
        "JSON envelope must not leak into the answer: {resp:?}"
    );
    assert_eq!(
        resp.citations,
        vec![real_id],
        "unwrapped citations validate against retrieved ids: {resp:?}"
    );
}

#[tokio::test]
async fn reflect_citations_limited_to_read_observations() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let mk_engine = |llm: Arc<dyn LlmProvider>| {
        Engine::new(
            store.clone(),
            llm,
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(1_800_000_000_000)),
        )
    };
    // Seed two entities (2 facts each so consolidation synthesizes for both),
    // then consolidate to give each entity an observation.
    let seed = MockLlm::script(vec![
        facts_resp(&[
            ("Hana leads the design team.", "Hana"),
            ("Hana mentors junior designers.", "Hana"),
            ("Ivan runs the infrastructure group.", "Ivan"),
            ("Ivan automates deploys.", "Ivan"),
        ]),
        LlmResponse {
            content: Some(
                r#"{"observations":[{"text":"Leads and mentors.","evidence":[{"fact_index":0,"quote":""}]}]}"#
                    .into(),
            ),
            tool_calls: vec![],
        },
        LlmResponse {
            content: Some(
                r#"{"observations":[{"text":"Runs infrastructure.","evidence":[{"fact_index":0,"quote":""}]}]}"#
                    .into(),
            ),
            tool_calls: vec![],
        },
    ]);
    let seeder = mk_engine(Arc::new(seed));
    retain(
        &seeder,
        "cl",
        "Hana leads the design team. Hana mentors junior designers. \
         Ivan runs the infrastructure group. Ivan automates deploys.",
    )
    .await;
    seeder.consolidate_once("cl", 10).await.unwrap();

    let obs_id_for = |name: &str| {
        let ent = seeder
            .store
            .entities_for_bank("cl")
            .unwrap()
            .into_iter()
            .find(|x| x.name == name)
            .unwrap_or_else(|| panic!("no entity {name}"))
            .id;
        seeder
            .store
            .observations_for_entity(&ent)
            .unwrap()
            .first()
            .unwrap_or_else(|| panic!("no observations for {name}"))
            .id
            .clone()
    };
    let hana_obs = obs_id_for("Hana");
    let ivan_obs = obs_id_for("Ivan");

    // reflect reads only Hana's observations, then tries to cite both
    // Hana's and Ivan's; the never-read one must be filtered out.
    let script = MockLlm::script(vec![
        tool_call("1", "read_observation", json!({"entity": "Hana"})),
        tool_call(
            "2",
            "finish",
            json!({"answer": "Hana leads design.", "citations": [ivan_obs, hana_obs.clone()]}),
        ),
    ]);
    let e = mk_engine(Arc::new(script));
    let resp = e.reflect("cl", "What about Hana?").await.unwrap();
    assert_eq!(
        resp.citations,
        vec![hana_obs],
        "only observations actually read may be cited: {resp:?}"
    );
}

#[tokio::test]
async fn reflect_forms_and_stores_opinions() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        extraction_resp(),
        tool_call("1", "recall", json!({"query": "python"})),
        tool_call(
            "2",
            "finish",
            json!({"answer": "I think Python suits this project."}),
        ),
        // closing ask: plain prose, not finish-shaped — ignored, the
        // volunteered answer stands
        synthesis_resp("Not a finish call."),
        LlmResponse {
            content: Some(
                r#"{"opinions":[{"text":"I think Python suits this project.","confidence":0.8}]}"#
                    .into(),
            ),
            tool_calls: vec![],
        },
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(
        &e,
        "o",
        "The user asked about programming languages for data science.",
    )
    .await;
    let resp = e
        .reflect("o", "Which language should I use?")
        .await
        .unwrap();
    assert_eq!(resp.answer, "I think Python suits this project.");
    assert_eq!(resp.answer_path, AnswerPath::Loop);
    // Opinion formation is backgrounded off the response path: poll the
    // store until the spawned extraction lands (bounded, like the worker
    // tests).
    let mut ops = vec![];
    for _ in 0..200 {
        ops = e.store.list_opinions("o").unwrap();
        if ops.len() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(ops.len(), 1, "opinion must form in the background");
    assert!((ops[0].confidence - 0.8).abs() < 1e-6);
    assert!(ops[0].embedding.is_some());
}

/// Regression (fullstack smoke, gpt-oss-20b): models that batch `recall`
/// and `finish` into one turn used to have the `finish` dropped by the
/// first-call-only handler, burn every round on recalls, and fail with
/// "reflect produced no answer". The whole batch must execute: recall
/// runs, then finish records the answer — in a single round, with the
/// same-batch recall results citable.
#[tokio::test]
async fn reflect_completes_from_batched_tool_calls() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let seeder = Engine::new(
        store.clone(),
        Arc::new(MockLlm::script(vec![extraction_resp()])),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(&seeder, "batch", "Alice works at Google as an engineer.").await;
    let mem_id = seeder
        .store
        .list_memories("batch", 10)
        .unwrap()
        .into_iter()
        .map(|m| m.id)
        .next()
        .expect("retained memory");

    let script = MockLlm::script(vec![LlmResponse {
        content: None,
        tool_calls: vec![
            ToolCall {
                id: "1".into(),
                name: "recall".into(),
                arguments: json!({"query": "Alice"}),
            },
            ToolCall {
                id: "2".into(),
                name: "finish".into(),
                arguments: json!({"answer": "Alice works at Google.", "citations": [mem_id]}),
            },
        ],
    }]);
    let e = Engine::new(
        store,
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("batch", "What does Alice do?").await.unwrap();
    assert_eq!(resp.answer, "Alice works at Google.");
    assert_eq!(resp.answer_path, AnswerPath::Loop);
    assert_eq!(
        resp.rounds, 2,
        "batched recall+finish is one loop round + the closing ask"
    );
    assert!(
        resp.citations.contains(&mem_id),
        "recall in the same batch makes its memories citable: {resp:?}"
    );
}

/// Regression (fullstack smoke, gpt-oss-20b): a response with neither
/// content nor tool calls (gateway-stripped reasoning) used to kill the
/// question outright. One nudge must recover it.
#[tokio::test]
async fn reflect_nudges_past_empty_response() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        LlmResponse {
            content: Some("   ".into()), // whitespace-only counts as empty
            tool_calls: vec![],
        },
        tool_call("1", "finish", json!({"answer": "Recovered after a nudge."})),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("quiet", "Anybody there?").await.unwrap();
    assert_eq!(resp.answer, "Recovered after a nudge.");
    assert_eq!(resp.answer_path, AnswerPath::Loop);
    assert_eq!(resp.rounds, 3, "nudge round + finish round + closing ask");
}

/// Two consecutive empty responses no longer fail the question: the loop
/// hands off to the forced-done round and the final-synthesis call, which
/// composes an answer from whatever the tools gathered (here: nothing —
/// the synthesis still answers, from the no-data prompt).
#[tokio::test]
async fn reflect_synthesizes_after_consecutive_empty_responses() {
    let store = SqliteStore::open_in_memory().unwrap();
    // empty, empty (loop breaks), empty (forced-done), "Nothing known."
    // (synthesis), opinions.
    let script = MockLlm::script(vec![
        LlmResponse {
            content: None,
            tool_calls: vec![],
        },
        LlmResponse {
            content: None,
            tool_calls: vec![],
        },
        LlmResponse {
            content: None,
            tool_calls: vec![],
        },
        synthesis_resp("Nothing known."),
        LlmResponse {
            content: Some(r#"{"opinions":[]}"#.into()),
            tool_calls: vec![],
        },
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("silent", "q").await.unwrap();
    assert_eq!(resp.answer, "Nothing known.");
    assert_eq!(resp.answer_path, AnswerPath::Synthesis);
    assert!(resp.synthesized);
    assert!(resp.citations.is_empty());
}

fn facts_resp(facts: &[(&str, &str)]) -> LlmResponse {
    let facts: Vec<_> = facts
        .iter()
        .map(|(text, entity)| {
            json!({"text": text, "fact_type": "world", "entities": [{"name": entity, "etype": "person"}], "occurred_start": null, "occurred_end": null, "causal": []})
        })
        .collect();
    LlmResponse {
        content: Some(json!({"facts": facts}).to_string()),
        tool_calls: vec![],
    }
}

#[tokio::test]
async fn opinion_linked_to_entity_is_reinforced_by_consolidation() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        // retain #1: seed Alice + Google facts/entities
        extraction_resp(),
        // reflect: recall, finish with an opinion-bearing answer
        tool_call("1", "recall", json!({"query": "Alice"})),
        tool_call(
            "2",
            "finish",
            json!({"answer": "I think Alice is meticulous."}),
        ),
        // closing ask: plain prose, not finish-shaped — ignored
        synthesis_resp("Not a finish call."),
        // opinion extraction tags the entity the opinion is about
        LlmResponse {
            content: Some(
                r#"{"opinions":[{"text":"I think Alice is meticulous.","confidence":0.6,"entities":[{"name":"Alice","etype":"person"}]}]}"#.into(),
            ),
            tool_calls: vec![],
        },
        // retain #2 (later timestamp): new Alice facts dirty the entity
        facts_resp(&[
            ("Alice delivered the migration ahead of schedule.", "Alice"),
            ("Alice wrote thorough design docs.", "Alice"),
        ]),
        // consolidation: observation synthesis for Alice
        LlmResponse {
            content: Some(
                r#"{"observations":[{"text":"Summary: Alice delivers reliably.","evidence":[{"fact_index":0,"quote":""}]}]}"#.into(),
            ),
            tool_calls: vec![],
        },
        // consolidation: opinion reinforcement verdict
        LlmResponse {
            content: Some(
                r#"{"changed":true,"text":"I now think Alice is meticulous and reliable.","confidence":0.9}"#.into(),
            ),
            tool_calls: vec![],
        },
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(&e, "ri", "Alice works at Google as an engineer.").await;
    let resp = e
        .reflect("ri", "What do you think of Alice?")
        .await
        .unwrap();
    assert_eq!(resp.answer_path, AnswerPath::Loop);

    // The opinion forms in a background task; wait for it before the
    // scripted retain #2 below (they share one script queue).
    let mut ops = vec![];
    for _ in 0..200 {
        ops = e.store.list_opinions("ri").unwrap();
        if ops.len() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let alice = e
        .store
        .entities_for_bank("ri")
        .unwrap()
        .into_iter()
        .find(|x| x.name == "Alice")
        .expect("alice entity exists");
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].entity_ids, vec![alice.id.clone()]);
    assert!((ops[0].confidence - 0.6).abs() < 1e-6);
    assert!(ops[0].history.is_empty());

    e.retain(RetainRequest {
        bank_id: "ri".into(),
        content:
            "Alice delivered the migration ahead of schedule. Alice wrote thorough design docs."
                .into(),
        context: None,
        timestamp_ms: Some(1_800_003_600_000),
        ..Default::default()
    })
    .await
    .unwrap();

    e.consolidate_once("ri", 10).await.unwrap();

    let ops = e.store.list_opinions("ri").unwrap();
    assert_eq!(ops.len(), 1, "{ops:?}");
    assert_eq!(ops[0].text, "I now think Alice is meticulous and reliable.");
    assert!((ops[0].confidence - 0.9).abs() < 1e-6, "{:?}", ops[0]);
    assert_eq!(ops[0].history.len(), 1, "{:?}", ops[0]);
    assert!((ops[0].history[0].confidence - 0.6).abs() < 1e-6);
    assert_eq!(ops[0].history[0].reason, "reinforced by consolidation");
    assert_eq!(ops[0].entity_ids, vec![alice.id]);
    assert!(!e.store.observations_for_bank("ri").unwrap().is_empty());
}

#[tokio::test]
async fn opinion_without_entities_is_untouched_by_consolidation() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        facts_resp(&[
            ("Bob joined Acme as a designer.", "Bob"),
            ("Bob redesigned the Acme landing page.", "Bob"),
        ]),
        LlmResponse {
            content: Some(
                r#"{"observations":[{"text":"Summary: Bob designs at Acme.","evidence":[{"fact_index":0,"quote":""}]}]}"#.into(),
            ),
            tool_calls: vec![],
        },
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    e.retain(RetainRequest {
        bank_id: "uo".into(),
        content: "Bob joined Acme as a designer. Bob redesigned the Acme landing page.".into(),
        context: None,
        // later than the opinion so the recency filter alone would not spare it
        timestamp_ms: Some(1_800_003_600_000),
        ..Default::default()
    })
    .await
    .unwrap();

    let now = e.clock.now_ms();
    e.store
        .upsert_opinion(&Opinion {
            id: "op-unlinked".into(),
            bank_id: "uo".into(),
            text: "I think Bob is unreliable.".into(),
            confidence: 0.7,
            evidence: vec![],
            history: vec![],
            entity_ids: vec![],
            updated_at_ms: now,
            created_at_ms: now,
            embedding: None,
        })
        .unwrap();

    let updated = e.consolidate_once("uo", 10).await.unwrap();
    assert!(updated >= 1, "consolidation ran: {updated}");
    assert!(!e.store.observations_for_bank("uo").unwrap().is_empty());

    let ops = e.store.list_opinions("uo").unwrap();
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].text, "I think Bob is unreliable.");
    assert!((ops[0].confidence - 0.7).abs() < 1e-6);
    assert!(ops[0].history.is_empty(), "{:?}", ops[0]);
    assert_eq!(ops[0].updated_at_ms, now);
    assert!(ops[0].entity_ids.is_empty());
}

#[tokio::test]
async fn mental_model_refresh() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        extraction_resp(),
        LlmResponse {
            content: Some("Standing answer (demo): the bank is currently small.".into()),
            tool_calls: vec![],
        },
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(&e, "m", "The user prefers dark mode in editors.").await;
    let model = e
        .refresh_mental_model("m", "What are the user's preferences?", None)
        .await
        .unwrap();
    assert!(model.content.contains("Standing answer"));
    assert!(model.last_refreshed_ms.is_some());
    assert_eq!(e.store.list_mental_models("m").unwrap().len(), 1);
}

// --- RS-3: chunking, dedup, re-ask, retries ---

struct WordEstimator;
impl TokenEstimator for WordEstimator {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

fn chunked_engine(llm: Arc<dyn LlmProvider>, chunk_size: usize, overlap: usize) -> Engine {
    Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        llm,
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(WordEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        chunk_size_tokens: chunk_size,
        chunk_overlap_tokens: overlap,
        ..Default::default()
    })
}

fn causal_facts_resp(facts: &[(&str, &[(usize, &str)])]) -> LlmResponse {
    let facts: Vec<_> = facts
        .iter()
        .map(|(text, causal)| {
            let causal: Vec<_> = causal
                .iter()
                .map(|(i, r)| json!({"fact_index": i, "relation": r}))
                .collect();
            json!({"text": text, "fact_type": "world", "entities": [], "occurred_start": null, "occurred_end": null, "causal": causal})
        })
        .collect();
    LlmResponse {
        content: Some(json!({"facts": facts}).to_string()),
        tool_calls: vec![],
    }
}

fn assert_causal_edge(e: &Engine, bank: &str, a_id: &str, b_id: &str) {
    let edges = e.store.edges_for_bank(bank, 1000).unwrap();
    assert!(
        edges.iter().any(|x| x.kind == EdgeKind::Causal
            && x.a_id == a_id
            && x.b_id == b_id
            && x.label.as_deref() == Some("causes")),
        "causal edge {a_id}->{b_id} missing: {edges:?}"
    );
}

#[tokio::test]
async fn two_chunk_retain_builds_cross_chunk_causal_edge() {
    // 6 words, 4-token chunks, 1-token overlap -> exactly two chunks:
    // ["alpha beta gamma delta", "delta epsilon zeta"]
    let script = MockLlm::script(vec![
        causal_facts_resp(&[
            ("Alice joined Google as an engineer.", &[]),
            ("Alice works on search infrastructure.", &[]),
        ]),
        // overlap re-extracts chunk 1's second fact; the new fact causally
        // points at it via chunk-local index 0
        causal_facts_resp(&[
            ("Alice works on search infrastructure!", &[]),
            ("Alice chose the SRE team for the growth.", &[(0, "causes")]),
        ]),
    ]);
    let e = chunked_engine(Arc::new(script), 4, 1);
    let r = retain(&e, "cc", "alpha beta gamma delta epsilon zeta").await;
    assert_eq!(r.memories_stored, 3, "{r:?}");

    let mems = e.store.list_memories("cc", 100).unwrap();
    let id = |needle: &str| {
        mems.iter()
            .find(|m| m.text.contains(needle))
            .unwrap_or_else(|| panic!("no memory containing {needle}: {mems:?}"))
            .id
            .clone()
    };
    let alice_infra = id("search infrastructure");
    let alice_chose = id("chose the SRE team");
    // the causal ref declared against the overlap duplicate must land on the
    // deduped (kept) fact, not the dropped copy
    assert_causal_edge(&e, "cc", &alice_chose, &alice_infra);
}

#[tokio::test]
async fn overlap_duplicate_fact_stored_once() {
    let script = MockLlm::script(vec![
        causal_facts_resp(&[
            ("Cara leads the design team.", &[]),
            ("Cara uses Figma every day.", &[]),
        ]),
        // chunk 2 sees the overlap region and re-extracts chunk 1's fact
        causal_facts_resp(&[("Cara uses Figma every day!", &[])]),
    ]);
    let e = chunked_engine(Arc::new(script), 4, 1);
    let r = retain(&e, "dup", "alpha beta gamma delta epsilon zeta").await;
    assert_eq!(r.memories_stored, 2, "{r:?}");
    let mems = e.store.list_memories("dup", 100).unwrap();
    assert_eq!(
        mems.iter().filter(|m| m.text.contains("Figma")).count(),
        1,
        "{mems:?}"
    );
}

#[tokio::test]
async fn retain_retries_transient_llm_failure() {
    let inner = MockLlm::script(vec![causal_facts_resp(&[("Dana moved to Austin.", &[])])]);
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(FlakyLlm::new(Arc::new(inner), 1, 429)),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        retry_max_attempts: 3,
        retry_base_delay_ms: 1,
        ..Default::default()
    });
    let r = retain(&e, "rt", "Dana moved to Austin for the music scene.").await;
    assert_eq!(r.memories_stored, 1, "{r:?}");
}

/// Transport-level failures (connection dropped mid-body, timeout — the
/// "error decoding response body" reqwest surfaces) are as transient as a
/// 429: one dropped body must not kill a whole conversation's retain.
#[tokio::test]
async fn retain_retries_transport_failure() {
    let inner = MockLlm::script(vec![causal_facts_resp(&[("Gus moved to Denver.", &[])])]);
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(FlakyLlm::new_transport(Arc::new(inner), 2)),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        retry_max_attempts: 3,
        retry_base_delay_ms: 1,
        ..Default::default()
    });
    let r = retain(&e, "rtt", "Gus moved to Denver for the skiing.").await;
    assert_eq!(r.memories_stored, 1, "{r:?}");
}

#[tokio::test]
async fn retain_retries_transient_embed_failure() {
    let inner = MockLlm::script(vec![causal_facts_resp(&[("Eli moved to Boston.", &[])])]);
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(inner),
        Arc::new(FlakyEmbed::new(Arc::new(MockEmbed::default()), 2, 503)),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        retry_max_attempts: 3,
        retry_base_delay_ms: 1,
        ..Default::default()
    });
    let r = retain(&e, "re", "Eli moved to Boston for the universities.").await;
    assert_eq!(r.memories_stored, 1, "{r:?}");
}

#[tokio::test]
async fn retain_fails_fast_on_permanent_llm_error() {
    let inner = MockLlm::script(vec![causal_facts_resp(&[("Fay moved to Reno.", &[])])]);
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(FlakyLlm::new(Arc::new(inner), 1, 400)),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        retry_max_attempts: 3,
        retry_base_delay_ms: 1,
        ..Default::default()
    });
    let err = e
        .retain(RetainRequest {
            bank_id: "pf".into(),
            content: "Fay moved to Reno last winter.".into(),
            context: None,
            timestamp_ms: None,
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::LlmStatus { status: 400, .. }),
        "expected typed 400, got: {err:?}"
    );
}

#[tokio::test]
async fn reflect_retries_transient_llm_failure() {
    // No retain first: the reflect loop's first turn is then the first LLM
    // call, so the single flaky 429 lands on the reflect path (not retain's).
    let inner = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "anything"})),
        tool_call("2", "finish", json!({"answer": "Fine."})),
    ]);
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(FlakyLlm::new(Arc::new(inner), 1, 429)),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        retry_max_attempts: 3,
        retry_base_delay_ms: 1,
        ..Default::default()
    });
    let resp = e.reflect("rtr", "How are things?").await.unwrap();
    assert_eq!(resp.answer, "Fine.");
}

#[tokio::test]
async fn reflect_retries_transient_recall_embed_failure() {
    // Same shape, but the flaky failure lands on recall's query-embed call
    // triggered by the reflect loop's `recall` tool turn.
    let inner = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "anything"})),
        tool_call("2", "finish", json!({"answer": "Fine."})),
    ]);
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(inner),
        Arc::new(FlakyEmbed::new(Arc::new(MockEmbed::default()), 1, 429)),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        retry_max_attempts: 3,
        retry_base_delay_ms: 1,
        ..Default::default()
    });
    let resp = e.reflect("rte", "How are things?").await.unwrap();
    assert_eq!(resp.answer, "Fine.");
}

#[tokio::test]
async fn retain_reasks_at_half_size_on_parse_failure() {
    let script = MockLlm::script(vec![
        // first attempt: unparseable (truncated JSON-mode output)
        LlmResponse {
            content: Some(r#"{"facts":[{"text":"Bob mo"#.into()),
            tool_calls: vec![],
        },
        // re-ask on half 1
        causal_facts_resp(&[("Bob moved to Austin last month.", &[])]),
        // re-ask on half 2: local causal index 0 must offset to global 1
        causal_facts_resp(&[
            ("Bob bought a house there.", &[]),
            ("Bob bought it because prices fell.", &[(0, "causes")]),
        ]),
    ]);
    let e = chunked_engine(Arc::new(script), 3000, 200);
    let r = retain(&e, "rh", "Bob moved recently and bought property.").await;
    assert_eq!(r.memories_stored, 3, "{r:?}");
    assert_eq!(r.reasks, 1, "one half-size re-ask: {r:?}");

    let mems = e.store.list_memories("rh", 100).unwrap();
    let id = |needle: &str| {
        mems.iter()
            .find(|m| m.text.contains(needle))
            .unwrap_or_else(|| panic!("no memory containing {needle}: {mems:?}"))
            .id
            .clone()
    };
    let bought = id("bought a house");
    let because = id("because prices fell");
    assert_causal_edge(&e, "rh", &because, &bought);
}

// --- RS-4: reranking -------------------------------------------------------

/// Fixture reranker with a known relevance ground truth: docs mentioning
/// "hiking" score highest, then "gear" docs, then everything else. Used to
/// prove the recall pipeline orders results by reranker score rather than
/// raw RRF order.
struct FixtureRerank;

#[async_trait]
impl RerankProvider for FixtureRerank {
    async fn rerank(
        &self,
        _query: &str,
        docs: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let score = |d: &str| {
            if d.contains("hiking") {
                5.0
            } else if d.contains("gear") {
                3.0
            } else {
                0.5
            }
        };
        let mut ranked: Vec<(usize, f32)> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, score(d)))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(top_k);
        Ok(ranked)
    }
}

const RERANK_FIXTURE: &str = "Alice said her favorite outdoor activity is a closely kept secret. \
     Alice and Bob bought outdoor gear for their favorite trails. \
     Alice loves hiking in the mountains. \
     The committee meeting was postponed until Friday. \
     Fireflies lit up the garden at dusk.";

fn rerank_fixture_engine(rerank: Arc<dyn RerankProvider>) -> Engine {
    Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(MockLlm::demo()),
        Arc::new(MockEmbed::default()),
        rerank,
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
}

async fn recall_fixture(e: &Engine) -> Vec<MemoryRef> {
    retain(e, "rr", RERANK_FIXTURE).await;
    e.recall(RecallRequest {
        bank_id: "rr".into(),
        query: "What is Alice's favorite outdoor activity?".into(),
        token_budget: 1000,
        top_k_per_arm: 10,
        kinds: None,
        ..Default::default()
    })
    .await
    .unwrap()
    .memories
}

/// With a real (scoring) reranker, the doc the reranker judges most relevant
/// comes first even though RRF ranked distractors above it, and rerank
/// scores propagate to the response — normalized (logits go through a
/// sigmoid) and modulated by the combined-scoring boosts.
#[tokio::test]
async fn reranked_recall_puts_relevant_doc_first() {
    let e = rerank_fixture_engine(Arc::new(FixtureRerank));
    let memories = recall_fixture(&e).await;
    assert!(memories.len() >= 3, "{memories:?}");
    assert!(
        memories[0].text.contains("hiking"),
        "top memory was: {:?}",
        memories[0]
    );
    assert!(
        memories[0].score > memories[1].score,
        "combined scores must order the response: {memories:?}"
    );
    // raw 5.0 logits → sigmoid ≈ 0.9933, times near-neutral boosts (all
    // fixture memories share the clock, so recency is uniform)
    assert!(
        memories[0].score > 0.99 && memories[0].score <= 1.0 * 1.1,
        "{memories:?}"
    );
    // known relevance order: hiking doc, then the gear doc, then the rest
    let pos = |needle: &str| {
        memories
            .iter()
            .position(|m| m.text.contains(needle))
            .unwrap_or_else(|| panic!("doc {needle} missing: {memories:?}"))
    };
    assert!(pos("hiking") < pos("gear"), "{memories:?}");
    assert!(pos("gear") < pos("postponed"), "{memories:?}");
    assert!(pos("gear") < pos("Fireflies"), "{memories:?}");
}

/// The control: NoopRerank keeps raw RRF order, in which keyword-sharing
/// distractors outrank the actually relevant doc — exactly the accuracy gap
/// RS-4's cross-encoder reranking closes.
#[tokio::test]
async fn noop_rerank_leaves_raw_rrf_order() {
    let e = rerank_fixture_engine(Arc::new(NoopRerank));
    let memories = recall_fixture(&e).await;
    assert!(memories.len() >= 3, "{memories:?}");
    assert!(
        !memories[0].text.contains("hiking"),
        "NoopRerank must preserve RRF order, but hiking doc won: {:?}",
        memories[0]
    );
    // the relevant doc is still retrieved, just not first
    assert!(
        memories.iter().any(|m| m.text.contains("hiking")),
        "{memories:?}"
    );
}

/// RS-5: retain now sends the vendored upstream extraction prompts, whose
/// responses use upstream's schema — what/when/who/why dimensions, plain
/// string entities, fact_type world|assistant, causal_relations with
/// target_index. Proves the pipeline stores such responses end-to-end:
/// combined fact text, experience kind for assistant facts, entity
/// resolution from string entities, and causal edges from target_index.
#[tokio::test]
async fn retain_parses_upstream_shaped_extraction_response() {
    let store = SqliteStore::open_in_memory().unwrap();
    let upstream_resp = LlmResponse {
        content: Some(
            json!({"facts": [
                {
                    "what": "Alice joined Google.",
                    "when": "Monday, September 14, 2026",
                    "where": "N/A",
                    "who": "Alice",
                    "why": "N/A",
                    "fact_kind": "event",
                    "occurred_start": "2026-09-14",
                    "occurred_end": null,
                    "fact_type": "world",
                    "entities": ["Alice", "Google"],
                    "causal_relations": null
                },
                {
                    "what": "I recommended Alice for the infra team.",
                    "when": "N/A",
                    "where": "N/A",
                    "who": "N/A",
                    "why": "She asked me which team to join.",
                    "fact_kind": "conversation",
                    "occurred_start": null,
                    "occurred_end": null,
                    "fact_type": "assistant",
                    "entities": ["Alice"],
                    "causal_relations": [{"target_index": 0, "relation_type": "caused_by"}]
                }
            ]})
            .to_string(),
        ),
        tool_calls: vec![],
    };
    let e = Engine::new(
        Arc::new(store),
        Arc::new(MockLlm::script(vec![upstream_resp])),
        Arc::new(MockEmbed::default()),
        Arc::new(NoopRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let r = e
        .retain(RetainRequest {
            bank_id: "upstream-shape".into(),
            content: "Alice joined Google. I recommended Alice for the infra team.".into(),
            context: None,
            timestamp_ms: None,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(r.memories_stored, 2, "{r:?}");

    let memories = e.store.recent_memories("upstream-shape", 10).unwrap();
    assert_eq!(memories.len(), 2, "{memories:?}");
    let joined = memories
        .iter()
        .find(|m| m.text.contains("Alice joined Google"))
        .expect("world fact stored with combined text");
    assert!(
        joined.text.contains("When: Monday, September 14, 2026"),
        "combined dimension text missing: {joined:?}"
    );
    assert!(joined.occurred_start_ms.is_some(), "{joined:?}");
    let recommended = memories
        .iter()
        .find(|m| m.text.contains("recommended Alice"))
        .expect("assistant fact stored");
    assert_eq!(recommended.kind, MemoryKind::Experience);

    // entity resolution turned the plain-string entities into canonical ones
    assert!(r.entities.len() >= 2, "{r:?}");

    // causal_relations[target_index=0] produced a Causal edge 1 -> 0
    let causal: Vec<_> = e
        .store
        .edges_for_bank("upstream-shape", 100)
        .unwrap()
        .into_iter()
        .filter(|ed| ed.kind == EdgeKind::Causal)
        .collect();
    assert_eq!(causal.len(), 1, "{causal:?}");
    assert_eq!(causal[0].b_id, joined.id);
    assert_eq!(causal[0].label.as_deref(), Some("caused_by"));
}

/// CR-4: an LLM that stalls mid-consolidation must hit the
/// `consolidate_timeout_secs` wall instead of holding the single-flight
/// consolidation lock forever. Delegates the first `stall_after` calls to
/// the demo mock, then never resolves on every later call.
struct StallAfterCallsLlm {
    inner: MockLlm,
    stall_after: usize,
    calls: std::sync::Mutex<usize>,
}

#[async_trait]
impl LlmProvider for StallAfterCallsLlm {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let call = {
            let mut c = self.calls.lock().unwrap();
            *c += 1;
            *c
        };
        if call > self.stall_after {
            std::future::pending::<()>().await;
            unreachable!("stalled LLM call must be cancelled by the consolidation timeout");
        }
        self.inner.complete(req).await
    }
}

#[tokio::test]
async fn consolidation_pass_is_bounded_by_timeout() {
    let store = SqliteStore::open_in_memory().unwrap();
    let llm = Arc::new(StallAfterCallsLlm {
        inner: MockLlm::demo(),
        // call 1 = retain extraction; call 2 = consolidation observation
        // synthesis, which must stall into the timeout
        stall_after: 1,
        calls: std::sync::Mutex::new(0),
    });
    let e = Engine::new(
        Arc::new(store),
        llm,
        Arc::new(MockEmbed::default()),
        Arc::new(NoopRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        consolidate_timeout_secs: 1,
        ..Default::default()
    });
    retain(
        &e,
        "t",
        "Alice joined Google. Alice works on search infrastructure.",
    )
    .await;

    let err = e.consolidate_once("t", 10).await.unwrap_err();
    assert!(err.to_string().contains("timed out"), "{err:?}");

    // dirty rows survive the aborted pass so the next tick retries them
    let dirty = e.store.list_dirty_entities("t", 10).unwrap();
    assert!(
        !dirty.is_empty(),
        "aborted pass must leave dirty rows intact: {dirty:?}"
    );
}

// --- consolidation partial progress: per-entity isolation + soft deadline ---

/// Routes consolidation synthesis responses by entity name (the synthesis
/// user prompt starts "ENTITY: <name>") and delegates everything else —
/// extraction, reflect — to the demo mock. Lets a test poison one entity's
/// response without depending on dirty-queue ordering or pass concurrency.
struct PerEntityLlm {
    demo: MockLlm,
    synthesis: std::sync::Mutex<HashMap<String, VecDeque<String>>>,
}

impl PerEntityLlm {
    fn new(synthesis: Vec<(&str, Vec<&str>)>) -> Self {
        Self {
            demo: MockLlm::demo(),
            synthesis: std::sync::Mutex::new(
                synthesis
                    .into_iter()
                    .map(|(name, responses)| {
                        (
                            name.to_string(),
                            responses.into_iter().map(String::from).collect(),
                        )
                    })
                    .collect(),
            ),
        }
    }
}

fn synthesis_entity(req: &LlmRequest) -> String {
    req.messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default()
        .strip_prefix("ENTITY: ")
        .and_then(|rest| rest.split('\n').next())
        .unwrap_or_default()
        .to_string()
}

#[async_trait]
impl LlmProvider for PerEntityLlm {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let system = req.system.clone().unwrap_or_default();
        if system.contains("You are a memory consolidation system") {
            let entity = synthesis_entity(req);
            let content = {
                let mut map = self.synthesis.lock().unwrap();
                map.get_mut(&entity)
                    .and_then(|q| q.pop_front())
                    .unwrap_or_else(|| "{}".into())
            };
            return Ok(LlmResponse {
                content: Some(content),
                tool_calls: vec![],
            });
        }
        self.demo.complete(req).await
    }
}

/// One entity's malformed synthesis response must cost only that entity:
/// its neighbors still consolidate and clear their dirty rows in the same
/// pass, the failed entity stays dirty, and the next pass retries it
/// successfully. (The old all-or-nothing pass aborted on the first parse
/// failure, which is what silently killed the 21-hour LongMemEval run.)
#[tokio::test]
async fn consolidation_isolates_per_entity_failures() {
    let llm = Arc::new(PerEntityLlm::new(vec![
        (
            "Alice",
            vec![
                r#"{"observations":[{"text":"Alice adopted a cat and feeds it salmon.","evidence":[{"fact_index":0,"quote":""}]}]}"#,
            ],
        ),
        (
            "Whiskers",
            vec![
                r#"{"observations":[{"text":"Whiskers is Alice's cat.","evidence":[{"fact_index":0,"quote":""}]}]}"#,
            ],
        ),
        (
            "Bob",
            vec![
                "not json at all",
                r#"{"observations":[{"text":"Bob plays synthesizer.","evidence":[{"fact_index":0,"quote":""}]}]}"#,
            ],
        ),
        (
            "Cara",
            vec![
                r#"{"observations":[{"text":"Cara leads and mentors the design team.","evidence":[{"fact_index":0,"quote":""}]}]}"#,
            ],
        ),
    ]));
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        llm,
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(
        &e,
        "iso",
        "Alice adopted a cat named Whiskers. Alice feeds Whiskers salmon every morning. \
         Bob bought a vintage synthesizer. Bob practices the synthesizer daily. \
         Cara leads the design team. Cara mentors three designers.",
    )
    .await;

    let updated = e.consolidate_once("iso", 10).await.unwrap();
    assert!(updated >= 3, "updated={updated}");

    // everyone except Bob cleared their dirty rows in the same pass
    let dirty = e.store.list_dirty_entities("iso", 10).unwrap();
    assert_eq!(dirty.len(), 1, "{dirty:?}");
    let bob = e
        .store
        .entities_for_bank("iso")
        .unwrap()
        .into_iter()
        .find(|x| x.name == "Bob")
        .expect("bob entity exists");
    assert_eq!(dirty[0], bob.id, "{dirty:?}");

    // Alice and Cara got observations despite Bob's malformed response
    let texts: Vec<String> = e
        .store
        .observations_for_bank("iso")
        .unwrap()
        .iter()
        .map(|o| o.text.clone())
        .collect();
    assert!(texts.iter().any(|t| t.contains("Alice")), "{texts:?}");
    assert!(texts.iter().any(|t| t.contains("Cara")), "{texts:?}");

    // the retry pass completes Bob with the queued recovery response
    let updated = e.consolidate_once("iso", 10).await.unwrap();
    assert!(updated >= 1, "updated={updated}");
    assert!(e.store.list_dirty_entities("iso", 10).unwrap().is_empty());
    let bob_obs = e.store.observations_for_entity(&bob.id).unwrap();
    assert!(!bob_obs.is_empty(), "{bob_obs:?}");
}

/// Delays the named entity's synthesis response; everything else answers
/// instantly via the demo mock.
struct DelayedEntityLlm {
    demo: MockLlm,
    delay_ms: HashMap<String, u64>,
}

#[async_trait]
impl LlmProvider for DelayedEntityLlm {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let system = req.system.clone().unwrap_or_default();
        if system.contains("You are a memory consolidation system") {
            let entity = synthesis_entity(req);
            if let Some(ms) = self.delay_ms.get(&entity) {
                tokio::time::sleep(Duration::from_millis(*ms)).await;
            }
        }
        self.demo.complete(req).await
    }
}

/// The soft pass deadline stops *starting* new entities (they stay dirty
/// for the next pass) while an in-flight entity that crosses the budget
/// still lands — the hard wall at 2x the budget only cuts off calls that
/// never return at all. Paused tokio time makes the 1.2s delay free.
#[tokio::test(start_paused = true)]
async fn consolidation_deadline_stops_starting_new_entities() {
    let clock = Arc::new(FixedClock::new(1_800_000_000_000));
    let store = Arc::new(SqliteStore::open_in_memory_with_clock(clock.clone()).unwrap());
    let llm = Arc::new(DelayedEntityLlm {
        demo: MockLlm::demo(),
        delay_ms: [("Alice".to_string(), 1200)].into(),
    });
    let e = Engine::new(
        store,
        llm,
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        clock.clone(),
    )
    .with_config(EngineConfig {
        consolidate_timeout_secs: 1,
        consolidate_concurrency: 1,
        ..Default::default()
    });

    // Two retains at distinct store-clock timestamps force a deterministic
    // dirty order: Alice's entities (T0) precede Bob's (T1). The store's
    // marked_at_ms comes from the injected clock, so it must be advanced
    // between retains exactly the way the eval harness does.
    let t0 = 1_800_000_000_000i64;
    clock.set(t0);
    e.retain(RetainRequest {
        bank_id: "dl".into(),
        content: "Alice joined Google. Alice works on Search infrastructure.".into(),
        context: None,
        timestamp_ms: Some(t0),
        ..Default::default()
    })
    .await
    .unwrap();
    clock.set(t0 + 3_600_000);
    e.retain(RetainRequest {
        bank_id: "dl".into(),
        content: "Bob bought a vintage synthesizer. Bob practices the synthesizer daily.".into(),
        context: None,
        timestamp_ms: Some(t0 + 3_600_000),
        ..Default::default()
    })
    .await
    .unwrap();

    // Alice's 1.2s synthesis crosses the 1s soft deadline but lands inside
    // the 2s hard wall; every entity queued behind her must be skipped,
    // not started — their dirty rows survive for a fresh pass.
    let updated = e.consolidate_once("dl", 10).await.unwrap();
    assert!(updated >= 1, "alice synthesized: {updated}");
    let dirty = e.store.list_dirty_entities("dl", 10).unwrap();
    assert!(
        !dirty.is_empty(),
        "entities after the deadline were started"
    );
    let bob = e
        .store
        .entities_for_bank("dl")
        .unwrap()
        .into_iter()
        .find(|x| x.name == "Bob")
        .expect("bob entity exists");
    assert!(
        dirty.contains(&bob.id),
        "bob must stay dirty after the deadline: {dirty:?}"
    );
    let alice = e
        .store
        .entities_for_bank("dl")
        .unwrap()
        .into_iter()
        .find(|x| x.name == "Alice")
        .expect("alice entity exists");
    assert!(
        !dirty.contains(&alice.id),
        "alice landed before the wall and must be cleared: {dirty:?}"
    );
    assert!(
        !e.store
            .observations_for_entity(&alice.id)
            .unwrap()
            .is_empty(),
        "alice's in-flight synthesis landed"
    );

    // The next pass gets a fresh budget and drains the rest.
    let updated = e.consolidate_once("dl", 10).await.unwrap();
    assert!(updated >= 1, "updated={updated}");
    assert!(e.store.list_dirty_entities("dl", 10).unwrap().is_empty());
}

// --- RS-13: upstream-parity entity resolution + recall pool width ----------

/// One upstream-shaped extraction fact for a scripted retain.
fn us_fact(what: &str, entities: &[&str]) -> serde_json::Value {
    json!({
        "what": what,
        "when": "N/A",
        "where": "N/A",
        "who": "N/A",
        "why": "N/A",
        "fact_kind": "event",
        "occurred_start": null,
        "occurred_end": null,
        "fact_type": "world",
        "entities": entities,
        "causal_relations": null,
    })
}

fn extraction_response(facts: &[serde_json::Value]) -> LlmResponse {
    LlmResponse {
        content: Some(json!({"facts": facts}).to_string()),
        tool_calls: vec![],
    }
}

async fn retain_at(e: &Engine, bank: &str, content: &str, ts: i64) -> RetainResponse {
    e.retain(RetainRequest {
        bank_id: bank.into(),
        content: content.into(),
        context: None,
        timestamp_ms: Some(ts),
        ..Default::default()
    })
    .await
    .unwrap()
}

/// The merge-positive case: a given-name mention whose fact shares TWO of
/// the full-name entity's co-occurring partners clears upstream's 0.6 bar
/// (0.5·0.667 name + damped co-occurrence + 1-day recency ≈ 0.72) and links
/// to the existing entity instead of fragmenting.
#[tokio::test]
async fn resolver_links_given_name_to_full_name_with_shared_partners() {
    let day = 86_400_000i64;
    let t0 = 1_800_000_000_000i64;
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(MockLlm::script(vec![
            extraction_response(&[
                us_fact(
                    "Alice Chen started working with the user at Globex on the Harbor project",
                    &["Alice Chen", "Globex", "Harbor project"],
                ),
                us_fact(
                    "Alice Chen gave a tour of the Globex office in Austin",
                    &["Alice Chen", "Globex", "Austin"],
                ),
            ]),
            extraction_response(&[us_fact(
                "Alice met the Harbor project team at Globex",
                &["Alice", "Globex", "Harbor project"],
            )]),
        ])),
        Arc::new(MockEmbed::default()),
        Arc::new(NoopRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(t0)),
    );

    retain_at(&e, "merge", "Alice Chen joined Globex.", t0).await;
    retain_at(&e, "merge", "Alice met the team.", t0 + day).await;

    let entities = e.store.entities_for_bank("merge").unwrap();
    let names: Vec<String> = entities.iter().map(|x| x.name.clone()).collect();
    assert!(
        names.iter().any(|n| n == "Alice Chen"),
        "canonical entity missing: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "Alice"),
        "given-name mention must link, not fragment: {names:?}"
    );
    let alice_chen = entities.iter().find(|x| x.name == "Alice Chen").unwrap();
    assert_eq!(
        e.store.find_by_alias("merge", "alice").unwrap().unwrap().id,
        alice_chen.id
    );
    let linked = e.store.facts_for_entity(&alice_chen.id, 10).unwrap();
    assert_eq!(linked.len(), 3, "all three facts link to one entity");
}

/// The fullstack-smoke knowledge-update shape. The job-change fact mentions
/// only "Alice" and shares ONE partner with the Alice Chen entity, so the
/// exact upstream score is 0.5·0.667 + (1/√3)/3·0.3 + 1-day recency ≈ 0.56
/// — below 0.6, and the person fragments exactly as upstream would. The
/// reflect-budget recall pool (300 per arm, upstream's fixed MID default)
/// must still surface the job-change fact for the "work now" query, which
/// is how upstream answers this question class despite fragmentation.
#[tokio::test]
async fn knowledge_update_survives_fragmentation_through_wide_recall_pool() {
    let day = 86_400_000i64;
    let t0 = 1_800_000_000_000i64;
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(MockLlm::script(vec![
            extraction_response(&[
                us_fact(
                    "User started working with Alice Chen on the Harbor project at Globex",
                    &["User", "Alice Chen", "Harbor project", "Globex"],
                ),
                us_fact(
                    "Alice Chen gave a tour of the Globex office in Austin",
                    &["Alice Chen", "Globex", "Austin"],
                ),
                us_fact(
                    "User studied for an exam and was quizzed by Alice Chen every Friday",
                    &["User", "Alice Chen"],
                ),
                us_fact(
                    "The Harbor project deadline moved to July 15",
                    &["Harbor project"],
                ),
            ]),
            extraction_response(&[
                us_fact(
                    "Alice left Globex and joined Anthropic as a research engineer",
                    &["Alice", "Globex", "Anthropic", "research engineer"],
                ),
                us_fact(
                    "User is concerned about the Harbor handover",
                    &["User", "Harbor project"],
                ),
            ]),
        ])),
        Arc::new(MockEmbed::default()),
        Arc::new(NoopRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(t0)),
    );

    retain_at(&e, "ku", "session one", t0).await;
    retain_at(&e, "ku", "session five", t0 + day).await;

    // The upstream-exact verdict: the person fragments — the given-name
    // mention's corroboration (one damped shared partner) is not enough.
    let names: Vec<String> = e
        .store
        .entities_for_bank("ku")
        .unwrap()
        .into_iter()
        .map(|x| x.name.clone())
        .collect();
    assert!(names.iter().any(|n| n == "Alice Chen"), "{names:?}");
    assert!(names.iter().any(|n| n == "Alice"), "{names:?}");
    let job_change = e
        .store
        .list_memories("ku", 50)
        .unwrap()
        .into_iter()
        .find(|m| m.text.contains("joined Anthropic"))
        .expect("job-change memory stored");

    // The reflect recall tool's pool width: upstream's fixed MID budget.
    let resp = e
        .recall(RecallRequest {
            bank_id: "ku".into(),
            query: "Today's date is 2027-01-16. Where does Alice Chen work now?".into(),
            token_budget: 2000,
            top_k_per_arm: e.config.recall_budget.resolve(None, 2000),
            kinds: None,
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(
        resp.memories.iter().any(|m| m.id == job_change.id),
        "job-change memory must reach the reflect context at the MID budget: {:?}",
        resp.memories
            .iter()
            .map(|m| m.text.clone())
            .collect::<Vec<_>>()
    );
}

// --- upstream consolidation response machine (creates/updates/deletes) ------

/// The upstream plan machine: pass 1 creates from `source_fact_ids`, pass 2
/// updates the observation by id (new text + merged evidence + history),
/// and a delete-with-merge folds a duplicate's evidence into its survivor
/// before removing it.
#[tokio::test]
async fn consolidation_applies_upstream_plan_creates_updates_merges() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let mk = |llm: Arc<dyn LlmProvider>| {
        Engine::new(
            store.clone(),
            llm,
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(1_800_000_000_000)),
        )
    };
    retain(
        &mk(Arc::new(MockLlm::script(vec![facts_resp(&[
            ("Alice owns a 2019 Honda Civic.", "Alice"),
            ("Alice commutes far.", "Alice"),
        ])]))),
        "plan",
        "Alice owns a 2019 Honda Civic. Alice commutes far.",
    )
    .await;
    let facts = store.list_memories("plan", 10).unwrap();
    let civic = facts
        .iter()
        .find(|m| m.text.contains("Civic"))
        .expect("civic fact")
        .id
        .clone();

    // Pass 1: upstream-shaped create.
    let pass1 = MockLlm::script(vec![LlmResponse {
        content: Some(
            json!({
                "creates": [{"text": "Alice owns a 2019 Honda Civic.",
                             "source_fact_ids": [civic], "reason": "new facet"}],
                "updates": [], "deletes": []
            })
            .to_string(),
        ),
        tool_calls: vec![],
    }]);
    mk(Arc::new(pass1))
        .consolidate_once("plan", 10)
        .await
        .unwrap();
    let obs = store.observations_for_bank("plan").unwrap();
    assert_eq!(obs.len(), 1);
    let obs_id = obs[0].id.clone();
    assert_eq!(obs[0].evidence.len(), 1);
    assert!(obs[0].embedding.is_some(), "{:?}", obs[0]);

    // A second entity synthesizes a near-verbatim duplicate observation.
    let comm = facts
        .iter()
        .find(|m| m.text.contains("commutes"))
        .expect("commute fact")
        .id
        .clone();
    retain(
        &mk(Arc::new(MockLlm::script(vec![facts_resp(&[
            ("Alice commutes far daily.", "Alice"),
            ("Alice drives to work.", "Alice"),
        ])]))),
        "plan",
        "Alice commutes far daily. Alice drives to work.",
    )
    .await;

    // Pass 2: an update that rewrites the observation text, plus a
    // delete-with-merge folding the duplicate into it.
    let dupe_text = "Alice owns a 2019 Honda Civic.";
    let dup_obs = Observation {
        id: "dup-obs".into(),
        bank_id: "plan".into(),
        entity_id: obs[0].entity_id.clone(),
        text: format!("{dupe_text} (duplicate)"),
        evidence: vec![],
        proof_count: 1,
        history: vec![],
        updated_at_ms: 1,
        created_at_ms: 1,
        embedding: None,
    };
    store.upsert_observation(&dup_obs).unwrap();
    // give the duplicate an embedding so only text changes are re-embedded
    store
        .set_observation_embedding("dup-obs", Some(vec![0.1; 512]))
        .unwrap();

    let pass2 = MockLlm::script(vec![LlmResponse {
        content: Some(
            json!({
                "creates": [],
                "updates": [{
                    "observation_id": obs_id,
                    "text": "Alice owns a 2019 Honda Civic and commutes far.",
                    "source_fact_ids": [comm],
                    "reason": "state change"
                }],
                "deletes": [{
                    "observation_id": "dup-obs",
                    "merged_into": obs_id,
                    "reason": "restates the Civic observation"
                }]
            })
            .to_string(),
        ),
        tool_calls: vec![],
    }]);
    mk(Arc::new(pass2))
        .consolidate_once("plan", 10)
        .await
        .unwrap();

    let obs = store.observations_for_bank("plan").unwrap();
    assert_eq!(obs.len(), 1, "merged duplicate removed: {obs:?}");
    assert_eq!(obs[0].id, obs_id);
    assert_eq!(
        obs[0].text,
        "Alice owns a 2019 Honda Civic and commutes far."
    );
    assert!(
        obs[0].text != *dupe_text || obs[0].history.is_empty(),
        "history recorded"
    );
    assert_eq!(
        obs[0].history,
        vec!["Alice owns a 2019 Honda Civic.".to_string()]
    );
    // update evidence (civic) + commute (comm) + duplicate's backfilled
    // first fact all merged into the survivor
    assert!(obs[0].evidence.len() >= 2, "{:?}", obs[0].evidence);
}

/// The bank-level exact-dupe guard: two passes whose model emits the same
/// create text write one observation whose evidence grows, not siblings.
#[tokio::test]
async fn consolidation_exact_dupe_guard_merges_repeated_creates() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let mk = |llm: Arc<dyn LlmProvider>| {
        Engine::new(
            store.clone(),
            llm,
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(1_800_000_000_000)),
        )
    };
    retain(
        &mk(Arc::new(MockLlm::script(vec![facts_resp(&[
            ("Alice joined Google.", "Alice"),
            ("Alice works on search.", "Alice"),
        ])]))),
        "dup",
        "Alice joined Google. Alice works on search.",
    )
    .await;
    let facts = store.list_memories("dup", 10).unwrap();
    let f0 = facts[0].id.clone();

    for pass in 0..2 {
        let script = MockLlm::script(vec![LlmResponse {
            content: Some(
                json!({
                    "creates": [{"text": "Alice works at Google on search.",
                                 "source_fact_ids": [f0], "reason": "pass"}],
                    "updates": [], "deletes": []
                })
                .to_string(),
            ),
            tool_calls: vec![],
        }]);
        mk(Arc::new(script))
            .consolidate_once("dup", 10)
            .await
            .unwrap();
        let obs = store.observations_for_bank("dup").unwrap();
        assert_eq!(obs.len(), 1, "pass {pass}: {obs:?}");
        assert_eq!(obs[0].proof_count, 1, "same fact deduped by memory id");
    }
}

/// RS-19: entities with more facts than one window page through them all —
/// the old fixed 60-fact cap silently dropped older facts of big entities
/// so they never reached an observation.
#[tokio::test]
async fn consolidation_pages_past_the_fact_cap() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let mk = |llm: Arc<dyn LlmProvider>| {
        Engine::new(
            store.clone(),
            llm,
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(1_800_000_000_000)),
        )
        .with_config(EngineConfig {
            consolidate_max_facts_per_entity: 2,
            ..Default::default()
        })
    };
    // Retain 1 (earlier): two Bob facts; retain 2 (later): two more.
    retain(
        &mk(Arc::new(MockLlm::script(vec![facts_resp(&[
            ("Bob joined Stripe.", "Bob"),
            ("Bob works on payments.", "Bob"),
        ])]))),
        "page",
        "Bob joined Stripe. Bob works on payments.",
    )
    .await;
    let e = mk(Arc::new(MockLlm::script(vec![
        facts_resp(&[
            ("Bob moved to the infrastructure team.", "Bob"),
            ("Bob was promoted to senior developer.", "Bob"),
        ]),
        // Window 1 = the two NEWEST facts (retain 2), window 2 = the older
        // pair; each window's plan cites its own facts by index.
        LlmResponse {
            content: Some(
                json!({"observations": [
                    {"text": "Bob moved to infrastructure and was promoted.",
                     "evidence": [{"fact_index": 0, "quote": ""}, {"fact_index": 1, "quote": ""}]}
                ]})
                .to_string(),
            ),
            tool_calls: vec![],
        },
        LlmResponse {
            content: Some(
                json!({"observations": [
                    {"text": "Bob joined Stripe and works on payments.",
                     "evidence": [{"fact_index": 0, "quote": ""}, {"fact_index": 1, "quote": ""}]}
                ]})
                .to_string(),
            ),
            tool_calls: vec![],
        },
    ])));
    e.retain(RetainRequest {
        bank_id: "page".into(),
        content: "Bob moved to the infrastructure team. Bob was promoted to senior developer."
            .into(),
        context: None,
        timestamp_ms: Some(1_800_086_400_000),
        ..Default::default()
    })
    .await
    .unwrap();
    e.consolidate_once("page", 10).await.unwrap();

    let obs = store.observations_for_bank("page").unwrap();
    assert_eq!(obs.len(), 2, "one observation per window: {obs:?}");
    let evidenced: HashSet<String> = obs
        .iter()
        .flat_map(|o| o.evidence.iter().map(|ev| ev.memory_id.clone()))
        .collect();
    let all_facts: HashSet<String> = store
        .list_memories("page", 100)
        .unwrap()
        .into_iter()
        .map(|m| m.id)
        .collect();
    assert_eq!(
        evidenced, all_facts,
        "every fact — including the older window's — must be consolidated"
    );
    assert!(obs.iter().all(|o| o.embedding.is_some()), "{obs:?}");
}

/// RS-19: the cosine near-dupe guard — a CREATE paraphrasing an existing
/// observation merges its evidence instead of writing a sibling (the
/// 2026-11-12 runs' paraphrase siblings, e.g. "Bob began his career at
/// Stripe…" vs "Bob started as a junior developer…"). A genuinely new
/// observation is untouched.
#[tokio::test]
async fn consolidation_cosine_guard_merges_paraphrase_siblings() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let mk = |llm: Arc<dyn LlmProvider>| {
        Engine::new(
            store.clone(),
            llm,
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(1_800_000_000_000)),
        )
    };
    // Pass 1: two facts → one observation, embedded.
    retain(
        &mk(Arc::new(MockLlm::script(vec![facts_resp(&[
            ("Bob joined Stripe on the payments team.", "Bob"),
            ("Bob started as a junior developer there.", "Bob"),
        ])]))),
        "cos",
        "Bob joined Stripe on the payments team. Bob started as a junior developer there.",
    )
    .await;
    let e = mk(Arc::new(MockLlm::script(vec![LlmResponse {
        content: Some(
            json!({"observations": [
                {"text": "Bob started as a junior developer on the payments team at Stripe.",
                 "evidence": [{"fact_index": 0, "quote": ""}, {"fact_index": 1, "quote": ""}]},
                {"text": "Bob keeps a saltwater aquarium with two clownfish.",
                 "evidence": [{"fact_index": 0, "quote": ""}]}
            ]})
            .to_string(),
        ),
        tool_calls: vec![],
    }])));
    e.consolidate_once("cos", 10).await.unwrap();
    let obs = store.observations_for_bank("cos").unwrap();
    assert_eq!(obs.len(), 2, "{obs:?}");

    // Pass 2: a new fact arrives; the model CREATEs a near-paraphrase of
    // the existing observation plus a genuinely new one.
    let e = mk(Arc::new(MockLlm::script(vec![
        facts_resp(&[("Bob was promoted in May.", "Bob")]),
        LlmResponse {
            content: Some(
                json!({"observations": [
                    // Paraphrase (adds two words — cosine ~0.93 under the
                    // bag-of-words mock embed): must merge, not sibling.
                    {"text": "Bob started as a junior developer on the payments team at Stripe in March.",
                     "evidence": [{"fact_index": 0, "quote": ""}]},
                    // Genuinely new topic: must create.
                    {"text": "Bob cycles to work most days.",
                     "evidence": [{"fact_index": 0, "quote": ""}]}
                ]})
                .to_string(),
            ),
            tool_calls: vec![],
        },
    ])));
    e.retain(RetainRequest {
        bank_id: "cos".into(),
        content: "Bob was promoted in May.".into(),
        context: None,
        timestamp_ms: Some(1_800_086_400_000),
        ..Default::default()
    })
    .await
    .unwrap();
    e.consolidate_once("cos", 10).await.unwrap();

    let obs = store.observations_for_bank("cos").unwrap();
    let texts: Vec<&str> = obs.iter().map(|o| o.text.as_str()).collect();
    assert!(
        !texts.contains(
            &"Bob started as a junior developer on the payments team at Stripe in March."
        ),
        "paraphrase must not write a sibling: {texts:?}"
    );
    assert!(
        texts.contains(&"Bob cycles to work most days."),
        "{texts:?}"
    );
    assert_eq!(obs.len(), 3, "2 originals + 1 genuinely new: {obs:?}");
    // The promoted fact's evidence merged into the paraphrase target.
    let promoted: Vec<&Observation> = obs
        .iter()
        .filter(|o| o.text.contains("junior developer"))
        .collect();
    assert_eq!(promoted.len(), 1, "{texts:?}");
    assert!(promoted[0].proof_count >= 3, "{:?}", promoted[0]);
}

// --- reflect: kind-separated tools, forced-done, split synthesis -------------

/// Records the final-synthesis user message and answers the loop from a
/// fixed script — lets tests inspect exactly what the answer machine saw.
struct CapturingLlm {
    loop_script: std::sync::Mutex<VecDeque<LlmResponse>>,
    synthesis_prompts: std::sync::Mutex<Vec<String>>,
}

#[async_trait]
impl LlmProvider for CapturingLlm {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let system = req.system.clone().unwrap_or_default();
        if system.contains("You MUST ONLY use information from retrieved tool results") {
            self.synthesis_prompts.lock().unwrap().push(
                req.messages
                    .last()
                    .map(|m| m.content.clone())
                    .unwrap_or_default(),
            );
            return Ok(synthesis_resp("Synthesized."));
        }
        if system.contains("You extract evidence from retrieved memory data") {
            return Ok(synthesis_resp("- (claims)"));
        }
        Ok(self
            .loop_script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(synthesis_resp(r#"{"facts": []}"#)))
    }
}

/// Upstream kind alignment: `recall` surfaces only world/experience facts
/// and `search_observations` only observations — the synthesis prompt sees
/// exactly the kind the called tool owns, never a mixed pool.
#[tokio::test]
async fn reflect_tools_are_kind_separated() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let seeder = Engine::new(
        store.clone(),
        Arc::new(MockLlm::script(vec![
            facts_resp(&[
                ("Alice joined Google.", "Alice"),
                ("Alice works on search.", "Alice"),
            ]),
            LlmResponse {
                content: Some(
                    r#"{"observations":[{"text":"Alice works at Google on search.","evidence":[{"fact_index":0,"quote":""}]}]}"#
                        .into(),
                ),
                tool_calls: vec![],
            },
        ])),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(&seeder, "ks", "Alice joined Google. Alice works on search.").await;
    seeder.consolidate_once("ks", 10).await.unwrap();

    // recall sees only raw facts. The loop searches once then goes silent
    // (two empties end the loop, a third answers nothing in the forced-done
    // round), so the fallback synthesis runs and its prompt is captured.
    let empty = LlmResponse {
        content: None,
        tool_calls: vec![],
    };
    let capture = Arc::new(CapturingLlm {
        loop_script: std::sync::Mutex::new(
            vec![
                tool_call("1", "recall", json!({"query": "Alice"})),
                empty.clone(),
                empty.clone(),
                empty.clone(),
            ]
            .into(),
        ),
        synthesis_prompts: std::sync::Mutex::new(vec![]),
    });
    let e = Engine::new(
        store.clone(),
        capture.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("ks", "Where does Alice work?").await.unwrap();
    assert_eq!(resp.answer_path, AnswerPath::Synthesis);
    let prompts = capture.synthesis_prompts.lock().unwrap().clone();
    assert_eq!(prompts.len(), 1);
    let prompt = &prompts[0];
    assert!(prompt.contains("Alice joined Google"), "{prompt}");
    assert!(
        !prompt.contains("Alice works at Google on search"),
        "observations must not ride the recall tool: {prompt}"
    );

    // search_observations sees only observations
    let capture2 = Arc::new(CapturingLlm {
        loop_script: std::sync::Mutex::new(
            vec![
                tool_call("1", "search_observations", json!({"query": "Alice"})),
                empty.clone(),
                empty.clone(),
                empty,
            ]
            .into(),
        ),
        synthesis_prompts: std::sync::Mutex::new(vec![]),
    });
    let e2 = Engine::new(
        store,
        capture2.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e2.reflect("ks", "Where does Alice work?").await.unwrap();
    assert_eq!(resp.answer_path, AnswerPath::Synthesis);
    let prompts = capture2.synthesis_prompts.lock().unwrap().clone();
    let prompt = &prompts[0];
    assert!(
        prompt.contains("Alice works at Google on search"),
        "{prompt}"
    );
    assert!(
        !prompt.contains("\"Alice joined Google.\""),
        "raw facts must not ride search_observations: {prompt}"
    );
}

/// The 2026-11-11 parity-run 500s: a model that searches every round and
/// never finishes must still get an answer — the forced-done round asks
/// once more, and the synthesis call composes from the gathered history
/// regardless.
#[tokio::test]
async fn reflect_answers_after_round_exhaustion_without_finish() {
    let store = SqliteStore::open_in_memory().unwrap();
    // 8 recall rounds (the default max), then the forced-done round's
    // response (empty), then the synthesis answer, then opinions.
    let mut script: Vec<LlmResponse> = (0..8)
        .map(|i| tool_call(&format!("r{i}"), "recall", json!({"query": "everything"})))
        .collect();
    script.push(LlmResponse {
        content: None,
        tool_calls: vec![],
    });
    script.push(synthesis_resp(
        "Best-effort synthesis of everything gathered.",
    ));
    let e = Engine::new(
        Arc::new(store),
        Arc::new(MockLlm::script(script)),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("exhaust", "What is known?").await.unwrap();
    assert_eq!(resp.answer, "Best-effort synthesis of everything gathered.");
    assert_eq!(resp.answer_path, AnswerPath::Synthesis);
    assert_eq!(resp.rounds, 9, "8 loop rounds + the forced-done round");
}

/// Histories over `reflect_max_context_tokens` take upstream's split path:
/// chunks map to dated claims, then one reduce call synthesizes — evidence
/// is split, never dropped.
#[tokio::test]
async fn reflect_split_synthesis_maps_claims_and_reduces() {
    let store = SqliteStore::open_in_memory().unwrap();
    let empty = LlmResponse {
        content: None,
        tool_calls: vec![],
    };
    let script = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "Alice"})),
        empty.clone(),
        empty.clone(),
        empty,
        synthesis_resp("- Alice joined Google (mentioned_at: unknown; memory_ids: x)"),
        synthesis_resp("Reduced answer from claims."),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_config(EngineConfig {
        // Tiny budget: any recall history overflows it, but the 1024-token
        // split floor keeps a single chunk → one claims call + one reduce.
        reflect_max_context_tokens: 10,
        ..Default::default()
    });
    let resp = e.reflect("split", "Where does Alice work?").await.unwrap();
    assert_eq!(resp.answer, "Reduced answer from claims.");
    assert_eq!(resp.answer_path, AnswerPath::Synthesis);
}

/// Upstream #4567 sufficiency gate: a `finish` reporting "nothing is
/// known" (`answer_found: false`) with no `recall` yet is refused exactly
/// once and the model is nudged to recall; the second finish is honored
/// even if recall never ran.
#[tokio::test]
async fn reflect_refuses_premature_negative_finish_until_recall() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        tool_call("1", "search_observations", json!({"query": "Alice"})),
        tool_call(
            "2",
            "finish",
            json!({"answer": "Nothing is known about Alice.", "answer_found": false}),
        ),
        tool_call("3", "recall", json!({"query": "Alice"})),
        tool_call(
            "4",
            "finish",
            json!({"answer": "Nothing is known about Alice.", "answer_found": false}),
        ),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("gate", "What does Alice do?").await.unwrap();
    assert_eq!(resp.answer, "Nothing is known about Alice.");
    assert_eq!(resp.answer_path, AnswerPath::Loop);
    assert_eq!(
        resp.rounds, 5,
        "refused finish + recall + honored finish + the closing ask"
    );
}

/// The refusal is one-shot per reflect: without a recall in between, the
/// model's immediate second `finish` must be honored (never an infinite
/// refusal loop).
#[tokio::test]
async fn reflect_honors_repeated_negative_finish_without_recall() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        tool_call(
            "1",
            "finish",
            json!({"answer": "Nothing known.", "answer_found": false}),
        ),
        tool_call(
            "2",
            "finish",
            json!({"answer": "Nothing known.", "answer_found": false}),
        ),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("gate2", "What is known?").await.unwrap();
    assert_eq!(resp.answer, "Nothing known.");
    assert_eq!(
        resp.rounds, 3,
        "first finish refused, second honored, + the closing ask"
    );
}

/// Upstream precedence, mechanically: the closing ask (upstream
/// `_ask_for_done`) always runs after the loop stops and its finish wins
/// over the loop's volunteered one — the volunteered text survives only
/// when the ask itself fails. No standalone synthesis call is made when
/// either loop-side answer exists (the parity rerun's terse answers came
/// from synthesizing over a thin history while discarding the loop's
/// conversation-context answer).
#[tokio::test]
async fn reflect_closing_ask_wins_and_no_synthesis_runs() {
    let store = SqliteStore::open_in_memory().unwrap();
    let capture = Arc::new(CapturingLlm {
        loop_script: std::sync::Mutex::new(
            vec![
                tool_call("1", "search_observations", json!({"query": "Alice"})),
                tool_call("2", "finish", json!({"answer": "Alice works at Google."})),
                tool_call(
                    "3",
                    "finish",
                    json!({"answer": "Alice works at Google as an engineer on search."}),
                ),
            ]
            .into(),
        ),
        synthesis_prompts: std::sync::Mutex::new(vec![]),
    });
    let e = Engine::new(
        Arc::new(store),
        capture.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("pref", "What does Alice do?").await.unwrap();
    assert_eq!(
        resp.answer, "Alice works at Google as an engineer on search.",
        "the closing ask's answer replaces the volunteered one"
    );
    assert_eq!(resp.answer_path, AnswerPath::ForcedDone);
    assert_eq!(resp.rounds, 3, "search + volunteered finish + closing ask");
    assert!(
        capture.synthesis_prompts.lock().unwrap().is_empty(),
        "synthesis must not run when a loop-side answer exists"
    );
    assert_eq!(resp.synthesis_ms, None);
}

/// The volunteered finish survives as the fallback when the closing ask
/// yields nothing usable (here: plain prose, not finish-shaped).
#[tokio::test]
async fn reflect_volunteered_finish_is_the_closing_ask_fallback() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "Alice"})),
        tool_call("2", "finish", json!({"answer": "Volunteered answer."})),
        synthesis_resp("Just prose, not a finish call."),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("fb", "q").await.unwrap();
    assert_eq!(resp.answer, "Volunteered answer.");
    assert_eq!(resp.answer_path, AnswerPath::Loop);
}

/// A closing-ask answer that sanitizes to nothing (nothing but pasted
/// memory ids) falls through to the volunteered finish instead of failing
/// the reflect — the 2026-11-12 run's 2025-12 scenario lost a question to
/// an all-or-nothing answer path.
#[tokio::test]
async fn reflect_ids_only_closing_answer_falls_through_to_volunteered() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "Alice"})),
        tool_call(
            "2",
            "finish",
            json!({"answer": "Volunteered substantive answer."}),
        ),
        tool_call(
            "3",
            "finish",
            json!({"answer": "Citations\n- 2aeb87d1-9b14-48d7-98cd-d822bd5c7475"}),
        ),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("fall", "q").await.unwrap();
    assert_eq!(resp.answer, "Volunteered substantive answer.");
    assert_eq!(resp.answer_path, AnswerPath::Loop);
}

/// The closing ask caps its requested output (upstream
/// `reflect_max_completion_tokens`): without a cap, gateways allocate a
/// huge default reasoning/output budget for thinking models and the call
/// 400s on context length (2026-11-12: 52k input + 79k allocation vs the
/// 131k window). It also forces the `finish` tool (upstream
/// `tool_choice=done`).
#[tokio::test]
async fn reflect_closing_ask_caps_output_and_forces_finish() {
    use rustsight_core::traits::LlmRequest;

    #[derive(Default)]
    struct ReqCapture {
        reqs: std::sync::Mutex<Vec<LlmRequest>>,
        script: std::sync::Mutex<VecDeque<LlmResponse>>,
    }
    #[async_trait]
    impl LlmProvider for ReqCapture {
        async fn complete(&self, req: &LlmRequest) -> rustsight_core::Result<LlmResponse> {
            self.reqs.lock().unwrap().push(req.clone());
            Ok(self
                .script
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(synthesis_resp("{}")))
        }
    }

    let capture = Arc::new(ReqCapture {
        reqs: std::sync::Mutex::new(vec![]),
        script: std::sync::Mutex::new(
            vec![
                tool_call("1", "recall", json!({"query": "Alice"})),
                tool_call("2", "finish", json!({"answer": "done"})),
                tool_call("3", "finish", json!({"answer": "Closing answer."})),
            ]
            .into(),
        ),
    });
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        capture.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("cap", "q").await.unwrap();
    assert_eq!(resp.answer, "Closing answer.");
    let reqs = capture.reqs.lock().unwrap();
    // The closing ask is the request that forced `finish`.
    let closing = reqs
        .iter()
        .find(|r| r.tool_choice.as_deref() == Some("finish"))
        .expect("closing ask must force the finish tool");
    assert_eq!(
        closing.max_tokens,
        Some(8192),
        "closing ask must cap output (default config)"
    );
    // Upstream's reflect answer-length target (default 4096) rides the
    // closing ask as the ## Length directive — without it the same model
    // that writes gold's exhaustive answers writes one-liners.
    let closing_user = closing
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.as_str())
        .unwrap_or_default();
    assert!(
        closing_user.contains("approximately 4096 tokens"),
        "length directive missing from closing ask: {closing_user}"
    );
    // The loop turns stay uncapped (upstream caps only the answer machine).
    let loop_reqs: Vec<_> = reqs.iter().filter(|r| r.tool_choice.is_none()).collect();
    assert!(!loop_reqs.is_empty());
    assert!(loop_reqs.iter().all(|r| r.max_tokens.is_none()));
}

/// Extraction carries upstream's output cap, and the config can change it.
#[tokio::test]
async fn retain_extraction_caps_output() {
    use rustsight_core::traits::LlmRequest;

    #[derive(Default)]
    struct ReqCapture {
        caps: std::sync::Mutex<Vec<Option<u64>>>,
    }
    #[async_trait]
    impl LlmProvider for ReqCapture {
        async fn complete(&self, req: &LlmRequest) -> rustsight_core::Result<LlmResponse> {
            self.caps.lock().unwrap().push(req.max_tokens);
            Ok(extraction_resp())
        }
    }

    for (config, expected) in [
        (EngineConfig::default(), 64_000),
        (
            EngineConfig {
                retain_max_completion_tokens: Some(9_000),
                ..Default::default()
            },
            9_000,
        ),
    ] {
        let capture = Arc::new(ReqCapture::default());
        let e = Engine::new(
            Arc::new(SqliteStore::open_in_memory().unwrap()),
            capture.clone(),
            Arc::new(MockEmbed::default()),
            Arc::new(MockRerank),
            Arc::new(SimpleTokenEstimator),
            Arc::new(FixedClock::new(1_800_000_000_000)),
        )
        .with_config(config);
        retain(&e, "cap", "Alice joined Google.").await;
        assert_eq!(*capture.caps.lock().unwrap(), vec![Some(expected)]);
    }
}

/// gpt-oss harness noise (channel/constrain special tokens leaked through
/// the gateway as content) must never become the answer — the turn counts
/// as empty and the closing ask recovers (2026-11-12 parity run: one
/// question answered with `<|channel|>final <|constrain|>citations":…]}`).
#[tokio::test]
async fn reflect_rejects_harness_noise_content() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        LlmResponse {
            content: Some(
                "<|start|>assistant<|channel|>final <|constrain|>citations\":[\"2aeb87d1\"]}<|call|>"
                    .into(),
            ),
            tool_calls: vec![],
        },
        tool_call("1", "recall", json!({"query": "Alice"})),
        tool_call(
            "2",
            "finish",
            json!({"answer": "Recovered from harness noise."}),
        ),
        synthesis_resp("Ignored prose."),
    ]);
    let e = Engine::new(
        Arc::new(store),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("noise", "q").await.unwrap();
    assert_eq!(
        resp.answer, "Recovered from harness noise.",
        "channel-token garbage must not leak: {resp:?}"
    );
    assert!(!resp.answer.contains("<|"), "{resp:?}");
}

/// Answers that pasted bare memory ids / "Citations" sections inline
/// (2026-11-12 run) get them stripped post-hoc (upstream
/// `presenter.drop_unseen_ids`) and the harvested ids land in the
/// validated citation set.
#[tokio::test]
async fn reflect_strips_inline_memory_ids_and_harvests_citations() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let seeder = Engine::new(
        store.clone(),
        Arc::new(MockLlm::script(vec![extraction_resp()])),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    retain(&seeder, "ids", "Alice works at Google as an engineer.").await;
    let real_id = seeder
        .recall(RecallRequest {
            bank_id: "ids".into(),
            query: "Alice".into(),
            token_budget: 500,
            top_k_per_arm: 5,
            kinds: None,
            ..Default::default()
        })
        .await
        .unwrap()
        .memories[0]
        .id
        .clone();

    let script = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "Alice"})),
        tool_call(
            "2",
            "finish",
            json!({
                "answer": format!(
                    "Alice works at Google.\n\n**Citations**\n- {real_id}\n- 00000000-dead-beef-0000-000000000000"
                ),
                "citations": []
            }),
        ),
        synthesis_resp("Ignored prose."),
    ]);
    let e = Engine::new(
        store,
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e.reflect("ids", "What does Alice do?").await.unwrap();
    assert_eq!(resp.answer, "Alice works at Google.", "{resp:?}");
    assert!(
        !resp.answer.contains("Citations") && !resp.answer.contains(&real_id),
        "{resp:?}"
    );
    // The harvested retrieved id becomes a citation; the unretrieved
    // one is filtered by validation.
    assert_eq!(resp.citations, vec![real_id], "{resp:?}");
}

// ── Hindsight-parity surface: documents, tags, metadata, per-op LLMs ────

fn doc_request(bank: &str, doc: &str, content: &str) -> RetainRequest {
    RetainRequest {
        bank_id: bank.into(),
        content: content.into(),
        document_id: Some(doc.into()),
        ..Default::default()
    }
}

fn texts(e: &Engine, bank: &str) -> Vec<String> {
    let mut t: Vec<String> = e
        .store
        .list_memories(bank, 100)
        .unwrap()
        .into_iter()
        .map(|m| m.text)
        .collect();
    t.sort();
    t
}

#[tokio::test]
async fn document_replace_swaps_memories_and_append_extends() {
    let e = engine();
    let first = e
        .retain(RetainRequest {
            tags: vec!["topic:career".into()],
            metadata: Some([("type".to_string(), "note".to_string())].into()),
            context: Some("career".into()),
            ..doc_request("docs", "d1", "Alice joined Google.")
        })
        .await
        .unwrap();
    assert_eq!(first.document_id.as_deref(), Some("d1"));
    assert_eq!(texts(&e, "docs"), vec!["Alice joined Google"]);
    let stored = &e.store.list_memories("docs", 10).unwrap()[0];
    assert_eq!(stored.document_id.as_deref(), Some("d1"));
    assert_eq!(stored.tags, vec!["topic:career".to_string()]);
    assert_eq!(
        stored
            .metadata
            .as_ref()
            .and_then(|m| m.get("type"))
            .map(String::as_str),
        Some("note")
    );

    // Replace (the default mode): the old version's memories are gone.
    e.retain(doc_request("docs", "d1", "Alice moved to Figma."))
        .await
        .unwrap();
    assert_eq!(texts(&e, "docs"), vec!["Alice moved to Figma"]);
    let doc = e.store.get_document("docs", "d1").unwrap().unwrap();
    assert_eq!(doc.original_text, "Alice moved to Figma.");

    // Append: existing memories stay, the body grows, only the tail is new.
    let appended = e
        .retain(RetainRequest {
            update_mode: UpdateMode::Append,
            ..doc_request("docs", "d1", "Alice leads design systems.")
        })
        .await
        .unwrap();
    assert_eq!(appended.memories_stored, 1);
    assert_eq!(
        texts(&e, "docs"),
        vec!["Alice leads design systems", "Alice moved to Figma"]
    );
    let doc = e.store.get_document("docs", "d1").unwrap().unwrap();
    assert_eq!(
        doc.original_text,
        "Alice moved to Figma.\nAlice leads design systems."
    );
    assert_eq!(e.store.count_document_memories("docs", "d1").unwrap(), 2);

    // Deleting the document removes its memories too.
    assert_eq!(e.forget_document("docs", "d1").unwrap(), 2);
    assert!(texts(&e, "docs").is_empty());
    assert!(e.store.get_document("docs", "d1").unwrap().is_none());
}

#[tokio::test]
async fn identical_document_reretain_is_a_no_op() {
    let e = engine();
    let first = e
        .retain(doc_request("idem", "cfg", "Alice joined Google."))
        .await
        .unwrap();
    assert_eq!(first.memories_stored, 1);
    let again = e
        .retain(doc_request("idem", "cfg", "Alice joined Google."))
        .await
        .unwrap();
    assert_eq!(again.memories_stored, 0, "{again:?}");
    assert_eq!(texts(&e, "idem").len(), 1, "no duplicate memories");
}

#[tokio::test]
async fn retain_without_document_id_files_under_a_generated_document() {
    let e = engine();
    let r = retain(&e, "gen", "Alice joined Google.").await;
    let doc_id = r.document_id.expect("generated document id");
    let doc = e.store.get_document("gen", &doc_id).unwrap().unwrap();
    assert_eq!(doc.original_text, "Alice joined Google.");
    // Two plain retains of the same text are two documents (no dedup).
    let second = retain(&e, "gen", "Alice joined Google.").await;
    assert_ne!(second.document_id.as_deref(), Some(doc_id.as_str()));
    assert_eq!(texts(&e, "gen").len(), 2);
}

#[tokio::test]
async fn document_replace_strips_stale_observation_evidence() {
    let e = engine();
    e.retain(doc_request(
        "obsdoc",
        "d1",
        "Alice joined Google. Alice works on search infrastructure.",
    ))
    .await
    .unwrap();
    e.consolidate_once("obsdoc", 50).await.unwrap();
    assert!(!e.store.observations_for_bank("obsdoc").unwrap().is_empty());

    e.retain(doc_request("obsdoc", "d1", "Bob moved to Austin."))
        .await
        .unwrap();
    let live: std::collections::HashSet<String> = e
        .store
        .list_memories("obsdoc", 100)
        .unwrap()
        .into_iter()
        .map(|m| m.id)
        .collect();
    for o in e.store.observations_for_bank("obsdoc").unwrap() {
        for ev in &o.evidence {
            assert!(
                live.contains(&ev.memory_id),
                "observation cites a deleted memory: {o:?}"
            );
        }
    }
}

async fn tagged_recall(e: &Engine, bank: &str, tags: &[&str], mode: TagsMatch) -> Vec<String> {
    let mut t: Vec<String> = e
        .recall(RecallRequest {
            bank_id: bank.into(),
            query: "Where does everyone work?".into(),
            token_budget: 4096,
            top_k_per_arm: 300,
            kinds: Some(vec![MemoryKind::World, MemoryKind::Experience]),
            tags: Some(TagFilter {
                tags: tags.iter().map(|s| s.to_string()).collect(),
                mode,
                groups: vec![],
            }),
            now_ms: None,
        })
        .await
        .unwrap()
        .memories
        .into_iter()
        .map(|m| m.text)
        .collect();
    t.sort();
    t
}

#[tokio::test]
async fn recall_tag_scope_follows_upstream_match_modes() {
    let e = engine();
    for (content, tags) in [
        ("Alice joined Google.", vec!["user:a"]),
        ("Bob joined Stripe.", vec!["user:b"]),
        ("Carol joined Anthropic.", vec!["user:a", "user:b"]),
        ("Dave joined Figma.", vec![]),
    ] {
        e.retain(RetainRequest {
            bank_id: "tags".into(),
            content: content.into(),
            tags: tags.into_iter().map(String::from).collect(),
            ..Default::default()
        })
        .await
        .unwrap();
    }
    // any: OR match, untagged included.
    assert_eq!(
        tagged_recall(&e, "tags", &["user:a"], TagsMatch::Any).await,
        vec![
            "Alice joined Google",
            "Carol joined Anthropic",
            "Dave joined Figma"
        ]
    );
    // any_strict: untagged excluded.
    assert_eq!(
        tagged_recall(&e, "tags", &["user:a"], TagsMatch::AnyStrict).await,
        vec!["Alice joined Google", "Carol joined Anthropic"]
    );
    // all_strict: every tag required.
    assert_eq!(
        tagged_recall(&e, "tags", &["user:a", "user:b"], TagsMatch::AllStrict).await,
        vec!["Carol joined Anthropic"]
    );
    // exact: set equality; an empty scope selects untagged memories only.
    assert_eq!(
        tagged_recall(&e, "tags", &["user:b"], TagsMatch::Exact).await,
        vec!["Bob joined Stripe"]
    );
    assert_eq!(
        tagged_recall(&e, "tags", &[], TagsMatch::Exact).await,
        vec!["Dave joined Figma"]
    );
}

#[tokio::test]
async fn observations_inherit_evidence_tags_for_scoping() {
    let e = engine();
    e.retain(RetainRequest {
        bank_id: "obstags".into(),
        content: "Alice joined Google. Alice works on search infrastructure.".into(),
        tags: vec!["user:a".into()],
        ..Default::default()
    })
    .await
    .unwrap();
    e.consolidate_once("obstags", 50).await.unwrap();
    let recall = |tags: &'static [&'static str]| {
        let e = &e;
        async move {
            e.recall(RecallRequest {
                bank_id: "obstags".into(),
                query: "Alice".into(),
                token_budget: 4096,
                top_k_per_arm: 300,
                kinds: Some(vec![MemoryKind::Observation]),
                tags: Some(TagFilter {
                    tags: tags.iter().map(|s| s.to_string()).collect(),
                    mode: TagsMatch::AnyStrict,
                    groups: vec![],
                }),
                now_ms: None,
            })
            .await
            .unwrap()
            .memories
        }
    };
    let hit = recall(&["user:a"]).await;
    assert!(!hit.is_empty(), "observation visible in its facts' scope");
    assert_eq!(hit[0].tags, vec!["user:a".to_string()]);
    assert!(
        recall(&["user:b"]).await.is_empty(),
        "observation hidden from another scope"
    );
}

#[tokio::test]
async fn caller_supplied_entities_are_linked_to_extracted_facts() {
    let e = engine();
    e.retain(RetainRequest {
        bank_id: "ents".into(),
        content: "The launch slipped by two weeks.".into(),
        entities: vec![ExtractedEntity {
            name: "Project Atlas".into(),
            etype: EntityType::Concept,
        }],
        ..Default::default()
    })
    .await
    .unwrap();
    let atlas = e
        .store
        .entities_for_bank("ents")
        .unwrap()
        .into_iter()
        .find(|en| en.name == "Project Atlas")
        .expect("caller-supplied entity created");
    assert!(
        !e.store.facts_for_entity(&atlas.id, 10).unwrap().is_empty(),
        "the extracted fact is linked to the supplied entity"
    );
}

#[tokio::test]
async fn per_operation_llm_overrides_route_calls() {
    // The main LLM is the demo mock (it would extract "Alice joined
    // Google"); the retain override is scripted with a distinct fact, so
    // the stored text shows which model retain actually called.
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(MockLlm::demo()),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_retain_llm(Arc::new(MockLlm::script(vec![causal_facts_resp(&[(
        "Routed through the retain model.",
        &[],
    )])])));
    let r = retain(&e, "ops", "Alice joined Google.").await;
    assert_eq!(r.memories_stored, 1);
    assert_eq!(texts(&e, "ops"), vec!["Routed through the retain model."]);
    // Reflect has no override, so it falls back to the main (demo) LLM.
    assert!(e.reflect("ops", "Where does Alice work?").await.is_ok());
}

#[tokio::test]
async fn reflect_renders_structured_output_from_response_schema() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    let script = MockLlm::script(vec![
        // loop: volunteered finish, then the closing ask's finish
        tool_call(
            "1",
            "finish",
            json!({"answer": "Alice works at Google.", "citations": []}),
        ),
        tool_call(
            "2",
            "finish",
            json!({"answer": "Alice works at Google.", "citations": []}),
        ),
        // structured-output rendering (before background opinion extraction)
        LlmResponse {
            content: Some(r#"{"company": "Google"}"#.into()),
            tool_calls: vec![],
        },
    ]);
    let e = Engine::new(
        store,
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    let resp = e
        .reflect_with(
            "so",
            "Where does Alice work?",
            &ReflectOptions {
                response_schema: Some(
                    json!({"type": "object", "properties": {"company": {"type": "string"}}}),
                ),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(resp.answer, "Alice works at Google.");
    assert_eq!(resp.structured_output, Some(json!({"company": "Google"})));
    assert!(resp.structured_output_error.is_none());
}

#[tokio::test]
async fn reflect_budgeted_rounds_follow_upstream_multipliers() {
    let e = engine();
    assert_eq!(e.reflect_rounds_for_budget(rustsight_core::Budget::Low), 5);
    assert_eq!(e.reflect_rounds_for_budget(rustsight_core::Budget::Mid), 10);
    assert_eq!(
        e.reflect_rounds_for_budget(rustsight_core::Budget::High),
        20
    );
}

// ── RS-10a: Memory Defense ───────────────────────────────────────────────

fn set_defense(e: &Engine, bank: &str, action: &str, enabled: bool) {
    e.store
        .ensure_bank(&Bank {
            id: bank.into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .unwrap();
    e.store
        .set_bank_config(
            bank,
            &json!({"memory_defense": {
                "enabled": enabled,
                "rules": [{"on": "sensitive_data", "action": action}]
            }}),
        )
        .unwrap();
}

const GH_TOKEN: &str = "ghp_abcdefghijklmnopqrstuvwxyz0123456789";

/// Every place a retained secret could land: memory texts, the document
/// body, and the source chunks the memories point at.
fn stored_text(e: &Engine, bank: &str) -> String {
    let mut all = String::new();
    let memories = e.store.list_memories(bank, 1000).unwrap();
    for m in &memories {
        all.push_str(&m.text);
        all.push('\n');
    }
    for d in e.store.list_documents(bank, 100).unwrap() {
        all.push_str(&d.original_text);
        all.push('\n');
    }
    let ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
    let chunk_ids: Vec<String> = e
        .store
        .memory_chunk_ids(bank, &ids)
        .unwrap()
        .into_values()
        .collect();
    for c in e.store.get_chunks(bank, &chunk_ids).unwrap() {
        all.push_str(&c.text);
        all.push('\n');
    }
    all
}

#[tokio::test]
async fn memory_defense_redact_keeps_secrets_out_of_every_store() {
    let e = engine();
    set_defense(&e, "md", "redact", true);
    let r = retain(
        &e,
        "md",
        &format!("Alice joined Google. Alice's deploy token is {GH_TOKEN} for the CI."),
    )
    .await;
    assert!(r.memories_stored >= 1, "{r:?}");
    let report = r.defense.expect("redaction reported");
    assert_eq!(report.action, defense::DefenseAction::Redact);
    assert_eq!(report.matched_types, vec!["github_token"]);
    assert_eq!(report.hits[0].preview, "ghp_...6789");
    let stored = stored_text(&e, "md");
    assert!(!stored.contains(GH_TOKEN), "secret leaked: {stored}");
    assert!(stored.contains("[REDACTED:github_token]"), "{stored}");
}

#[tokio::test]
async fn memory_defense_block_refuses_and_stores_nothing() {
    let e = engine();
    set_defense(&e, "mb", "block", true);
    let err = e
        .retain(RetainRequest {
            bank_id: "mb".into(),
            content: format!("Alice pasted {GH_TOKEN} into chat."),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Blocked { .. }), "{err:?}");
    assert!(err.to_string().contains("github_token"), "{err}");
    assert!(e.store.list_memories("mb", 100).unwrap().is_empty());
    assert!(e.store.list_documents("mb", 100).unwrap().is_empty());
    // Clean content still goes through under the same policy.
    let r = retain(&e, "mb", "Alice joined Google.").await;
    assert!(r.memories_stored >= 1 && r.defense.is_none(), "{r:?}");
}

#[tokio::test]
async fn memory_defense_disabled_or_absent_is_a_no_op() {
    let e = engine();
    set_defense(&e, "off", "redact", false);
    let content = format!("Alice's token is {GH_TOKEN} today.");
    let r = retain(&e, "off", &content).await;
    assert!(r.defense.is_none());
    assert!(stored_text(&e, "off").contains(GH_TOKEN));
    let r = retain(&e, "none", &content).await;
    assert!(r.defense.is_none());
}

#[tokio::test]
async fn memory_defense_malformed_policy_fails_closed() {
    let e = engine();
    retain(&e, "bad", "Alice joined Google.").await;
    e.store
        .set_bank_config(
            "bad",
            &json!({"memory_defense": {"enabled": true, "rules": [{"on": "sensitive_data", "action": "nuke"}]}}),
        )
        .unwrap();
    let err = e
        .retain(RetainRequest {
            bank_id: "bad".into(),
            content: format!("token {GH_TOKEN}"),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, Error::BadInput(_)), "{err:?}");
    assert!(!stored_text(&e, "bad").contains(GH_TOKEN));
}

// ── RS-10b: lifecycle events ─────────────────────────────────────────────

#[derive(Default)]
struct CaptureSink(std::sync::Mutex<Vec<events::EngineEvent>>);

impl events::EventSink for CaptureSink {
    fn emit(&self, event: &events::EngineEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}

impl CaptureSink {
    fn of(&self, t: events::EventType) -> Vec<events::EngineEvent> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.event == t)
            .cloned()
            .collect()
    }
}

#[tokio::test]
async fn events_report_retain_consolidation_and_defense() {
    use events::{EventData, EventType};
    let script = MockLlm::script(vec![
        facts_resp(&[
            ("Alice joined Google as an engineer.", "Alice"),
            ("Alice works on search infrastructure.", "Alice"),
        ]),
        LlmResponse {
            content: Some(
                r#"{"observations":[{"text":"Alice works at Google on search.","evidence":[{"fact_index":0,"quote":""}]}]}"#
                    .into(),
            ),
            tool_calls: vec![],
        },
    ]);
    let sink = Arc::new(CaptureSink::default());
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        Arc::new(script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_event_sink(sink.clone());

    set_defense(&e, "ev", "redact", true);
    e.retain(RetainRequest {
        bank_id: "ev".into(),
        content: format!("Alice joined Google as an engineer. Token {GH_TOKEN}."),
        document_id: Some("doc-1".into()),
        tags: vec!["team".into()],
        operation_id: Some("op-1".into()),
        ..Default::default()
    })
    .await
    .unwrap();

    let retained = sink.of(EventType::RetainCompleted);
    assert_eq!(retained.len(), 1);
    let r = &retained[0];
    assert_eq!(
        (r.bank_id.as_str(), r.operation_id.as_str()),
        ("ev", "op-1")
    );
    assert_eq!(r.status, "completed");
    assert_eq!(
        r.data,
        EventData::Retain {
            document_id: Some("doc-1".into()),
            tags: Some(vec!["team".into()]),
            memory_unit_count: Some(2),
        }
    );
    let defended = sink.of(EventType::MemoryDefenseTriggered);
    assert_eq!(defended.len(), 1);
    assert_eq!(defended[0].status, "redact");
    assert_eq!(defended[0].operation_id, "op-1");
    let wire: serde_json::Value = serde_json::from_str(&defended[0].wire_json()).unwrap();
    assert_eq!(wire["event"], "memory_defense.triggered");
    assert_eq!(wire["data"]["document_id"], "doc-1");
    assert_eq!(wire["data"]["matched_types"], json!(["github_token"]));
    assert_eq!(wire["data"]["hits"][0]["preview"], "ghp_...6789");
    assert!(!defended[0].wire_json().contains(GH_TOKEN));

    // A pass with work is reported; a pass over the drained queue is not.
    assert_eq!(e.consolidate_once("ev", 10).await.unwrap(), 1);
    assert_eq!(e.consolidate_once("ev", 10).await.unwrap(), 0);
    let consolidated = sink.of(EventType::ConsolidationCompleted);
    assert_eq!(consolidated.len(), 1);
    assert_eq!(consolidated[0].status, "completed");
    assert_eq!(
        consolidated[0].data,
        EventData::Consolidation {
            observations_created: Some(1),
            observations_updated: Some(0),
            observations_deleted: Some(0),
            error_message: None,
        }
    );
}

#[tokio::test]
async fn blocked_retain_reports_defense_but_not_completion() {
    let sink = Arc::new(CaptureSink::default());
    let e = engine().with_event_sink(sink.clone());
    set_defense(&e, "evb", "block", true);
    let err = e
        .retain(RetainRequest {
            bank_id: "evb".into(),
            content: format!("token {GH_TOKEN}"),
            ..Default::default()
        })
        .await;
    assert!(err.is_err());
    assert!(sink.of(events::EventType::RetainCompleted).is_empty());
    let blocked = sink.of(events::EventType::MemoryDefenseTriggered);
    assert_eq!(blocked.len(), 1);
    assert_eq!(blocked[0].status, "block");
}

// ── RS-10c: attachments ──────────────────────────────────────────────────

#[tokio::test]
async fn attachments_die_with_their_document_and_bank() {
    use rustsight_core::files::Attachment;
    let e = engine();
    retain(&e, "att", "Alice joined Google.").await;
    let put = |doc: &str, bytes: &[u8]| {
        let a = Attachment::new("att", doc, "f.txt", Some("text/plain"), bytes.to_vec(), 1);
        e.store.put_attachment(&a).unwrap();
        a.id
    };
    let a = put("doc-a", b"one");
    let b = put("doc-b", b"two");
    // The same bytes in two documents: one id, held by both.
    let shared = put("doc-b", b"one");
    assert_eq!(a, shared);

    e.forget_document("att", "doc-a").unwrap();
    let survivor = e.store.get_attachment("att", &a).unwrap().unwrap();
    assert_eq!(survivor.document_id, "doc-b", "doc-b's copy outlives doc-a");
    assert_eq!(survivor.bytes, b"one");
    assert!(e.store.get_attachment("att", &b).unwrap().is_some());

    e.delete_bank("att").await.unwrap();
    assert!(e.store.get_attachment("att", &a).unwrap().is_none());
    assert!(e.store.get_attachment("att", &b).unwrap().is_none());
}

// ── RS-10d: bank templates ───────────────────────────────────────────────

/// Records every system prompt and answers in plain prose (the loop's
/// volunteered answer; the closing ask then yields to it).
#[derive(Default)]
struct SystemCapture(std::sync::Mutex<Vec<String>>);

#[async_trait]
impl LlmProvider for SystemCapture {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        self.0
            .lock()
            .unwrap()
            .push(req.system.clone().unwrap_or_default());
        Ok(LlmResponse {
            content: Some("Here is the answer.".into()),
            tool_calls: vec![],
        })
    }
}

fn manifest(v: serde_json::Value) -> templates::Manifest {
    templates::parse_manifest(&v).unwrap()
}

#[tokio::test]
async fn imported_directives_reach_the_reflect_prompt() {
    let llm = Arc::new(SystemCapture::default());
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        llm.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    );
    e.import_template(
        "dir",
        &manifest(json!({"version": "1", "directives": [
            {"name": "Formal", "content": "Answer formally.", "priority": 1},
            {"name": "Urgent", "content": "Lead with the deadline.", "priority": 5},
            {"name": "Paused", "content": "Speak in rhyme.", "is_active": false},
            {"name": "Sales", "content": "Mention the discount.", "tags": ["sales"]}
        ]})),
        false,
    )
    .unwrap();

    e.reflect("dir", "What should I do?").await.unwrap();
    let first = llm.0.lock().unwrap()[0].clone();
    let urgent = first
        .find("**Urgent**: Lead with the deadline.")
        .expect("urgent");
    let formal = first.find("**Formal**: Answer formally.").expect("formal");
    assert!(urgent < formal, "higher priority first");
    assert!(!first.contains("rhyme"), "inactive directives never apply");
    assert!(
        !first.contains("discount"),
        "tagged directives need a matching scope"
    );

    llm.0.lock().unwrap().clear();
    let scoped = ReflectOptions {
        tags: Some(TagFilter {
            tags: vec!["sales".into()],
            mode: TagsMatch::Any,
            groups: vec![],
        }),
        ..Default::default()
    };
    e.reflect_with("dir", "What should I do?", &scoped)
        .await
        .unwrap();
    let first = llm.0.lock().unwrap()[0].clone();
    assert!(
        first.contains("**Sales**: Mention the discount."),
        "{first}"
    );
    assert!(first.contains("**Formal**"), "untagged ones still apply");
}

#[tokio::test]
async fn default_template_lands_on_a_bank_created_by_retain() {
    let base = engine();
    let e = Engine::new(
        Arc::clone(&base.store),
        Arc::new(MockLlm::demo()),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(1_800_000_000_000)),
    )
    .with_default_template(manifest(json!({
        "version": "1",
        "bank": {"reflect_mission": "Support desk memory.", "disposition_empathy": 5},
        "mental_models": [{"id": "overview", "name": "Overview", "source_query": "What matters?"}],
        "directives": [{"name": "Kind", "content": "Be kind."}]
    })));
    retain(&e, "fresh", "Alice joined Google.").await;
    let bank = e.store.get_bank("fresh").unwrap().unwrap();
    assert_eq!(bank.mission.as_deref(), Some("Support desk memory."));
    assert_eq!(bank.disposition.empathy, 5);
    assert_eq!(e.store.list_directives("fresh").unwrap()[0].name, "Kind");
    let models = e.store.list_mental_models("fresh").unwrap();
    assert_eq!(models[0].id, "fresh/overview");
    assert_eq!(models[0].name.as_deref(), Some("Overview"));

    // Only new banks get it: a later change survives the next retain.
    e.update_bank_config(
        "fresh",
        &serde_json::from_value(json!({"reflect_mission": "Changed."})).unwrap(),
    )
    .unwrap();
    retain(&e, "fresh", "Bob joined Google.").await;
    let bank = e.store.get_bank("fresh").unwrap().unwrap();
    assert_eq!(bank.mission.as_deref(), Some("Changed."));
    assert_eq!(e.store.list_directives("fresh").unwrap().len(), 1);
}

// ── RS-10e: multilingual ─────────────────────────────────────────────────

/// Answers from a script and records each request's full prompt text.
struct PromptCapture {
    script: std::sync::Mutex<VecDeque<LlmResponse>>,
    prompts: std::sync::Mutex<Vec<String>>,
}

impl PromptCapture {
    fn new(script: Vec<LlmResponse>) -> Self {
        Self {
            script: std::sync::Mutex::new(script.into()),
            prompts: std::sync::Mutex::default(),
        }
    }
}

#[async_trait]
impl LlmProvider for PromptCapture {
    async fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let mut text = req.system.clone().unwrap_or_default();
        for m in &req.messages {
            text.push('\n');
            text.push_str(&m.content);
        }
        self.prompts.lock().unwrap().push(text);
        Ok(self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(synthesis_resp("{}")))
    }
}

const ZH_NOW: i64 = 1_795_000_000_000; // 2026-11-18 (a Wednesday)

#[tokio::test]
async fn chinese_content_keeps_its_script_and_is_searchable() {
    let llm = Arc::new(PromptCapture::new(vec![facts_resp(&[
        ("爱丽丝上周加入了谷歌，担任搜索工程师。", "爱丽丝"),
        ("Bob moved to Berlin.", "Bob"),
    ])]));
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        llm.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(ZH_NOW)),
    );
    retain(
        &e,
        "zh",
        "爱丽丝上周加入了谷歌，担任搜索工程师。Bob moved to Berlin.",
    )
    .await;

    // No output language: the source-language rule is in the prompt and
    // the stored fact and entity stay in native script.
    assert!(llm.prompts.lock().unwrap()[0].contains("Never translate."));
    let names: Vec<String> = e
        .store
        .entities_for_bank("zh")
        .unwrap()
        .into_iter()
        .map(|en| en.name)
        .collect();
    assert!(names.contains(&"爱丽丝".to_string()), "{names:?}");

    // A Chinese keyword reaches the fact through BM25 bigrams, and a
    // Latin word still finds the English fact.
    let hits = e
        .store
        .search_bm25("zh", &bm25_terms("谷歌工程师"), 10)
        .unwrap();
    let texts: Vec<String> = hits
        .iter()
        .map(|(id, _, _)| e.store.get_memory(id).unwrap().unwrap().text)
        .collect();
    assert_eq!(
        texts,
        vec!["爱丽丝上周加入了谷歌，担任搜索工程师。".to_string()]
    );
    let hits = e
        .store
        .search_bm25("zh", &bm25_terms("berlin"), 10)
        .unwrap();
    assert_eq!(hits.len(), 1);

    // 上周 narrows the temporal arm to last calendar week (Mon 9 – Sun 15).
    let r = e
        .recall(RecallRequest {
            bank_id: "zh".into(),
            query: "爱丽丝上周做了什么？".into(),
            token_budget: 1000,
            top_k_per_arm: 20,
            ..Default::default()
        })
        .await
        .unwrap();
    // 2026-11-09T00:00Z .. 2026-11-16T00:00Z
    assert_eq!(r.window, Some((1_794_182_400_000, 1_794_787_200_000)));
}

#[tokio::test]
async fn output_language_reaches_extraction_and_reflect() {
    let llm = Arc::new(PromptCapture::new(vec![facts_resp(&[(
        "Alice rejoint Google.",
        "Alice",
    )])]));
    let mut e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        llm.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(ZH_NOW)),
    );
    e.config.output_language = Some("French".into());
    retain(&e, "fr", "Alice joined Google.").await;
    let extraction = llm.prompts.lock().unwrap()[0].clone();
    assert!(
        extraction.contains("Respond exclusively in French"),
        "{extraction}"
    );
    assert!(
        !extraction.contains("Never translate."),
        "the source-language rule is dropped"
    );

    llm.prompts.lock().unwrap().clear();
    e.reflect("fr", "Where does Alice work?").await.unwrap();
    let first = llm.prompts.lock().unwrap()[0].clone();
    assert!(first.contains("Respond exclusively in French"), "{first}");
    assert!(!first.contains("## LANGUAGE RULE"));
}

// ── RS-10f: hierarchical config ──────────────────────────────────────────

#[tokio::test]
async fn tenant_chunk_size_applies_to_its_banks_and_a_bank_override_wins() {
    let llm = Arc::new(PromptCapture::new(vec![]));
    let e = Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        llm.clone(),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(ZH_NOW)),
    );
    let mut tenant = serde_json::Map::new();
    tenant.insert("chunk_size_tokens".into(), json!(40));
    tenant.insert("chunk_overlap_tokens".into(), json!(0));
    e.update_tenant_config("acme", &tenant).unwrap();
    let join = |bank: &str, extra: Option<(&str, serde_json::Value)>| {
        let mut updates = serde_json::Map::new();
        updates.insert("tenant".into(), json!("acme"));
        if let Some((k, v)) = extra {
            updates.insert(k.into(), v);
        }
        e.update_bank_config(bank, &updates).unwrap();
    };
    join("in-tenant", None);
    join("overridden", Some(("chunk_size_tokens", json!(100_000))));

    let content = (0..60)
        .map(|i| format!("Sentence number {i} says Alice visited place {i}."))
        .collect::<Vec<_>>()
        .join(" ");
    let calls = |bank: &'static str| {
        let content = content.clone();
        let llm = llm.clone();
        let e = &e;
        async move {
            llm.prompts.lock().unwrap().clear();
            retain(e, bank, &content).await;
            llm.prompts.lock().unwrap().len()
        }
    };
    let alone = calls("no-tenant").await;
    let tenant_calls = calls("in-tenant").await;
    let overridden = calls("overridden").await;
    assert_eq!(alone, 1, "global chunk size: one extraction");
    assert!(tenant_calls > 1, "tenant chunk size splits: {tenant_calls}");
    assert_eq!(overridden, 1, "the bank's own value beats the tenant");

    let r = e.resolved_config("overridden").unwrap();
    assert_eq!(r.source("chunk_size_tokens"), Some(Layer::Bank));
    assert_eq!(r.source("chunk_overlap_tokens"), Some(Layer::Tenant));
}

// --- RS-22: memory curation (edit / invalidate / restore) ------------------

/// Embedder that fails permanently (HTTP 400, never retried) while its
/// switch is on — proves curation makes its provider call before writing.
struct SwitchEmbed {
    inner: MockEmbed,
    fail: std::sync::atomic::AtomicBool,
}

#[async_trait]
impl EmbedProvider for SwitchEmbed {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if self.fail.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Error::EmbedStatus {
                status: 400,
                message: "switched off".into(),
            });
        }
        self.inner.embed(texts).await
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

const CURATE_NOW: i64 = 1_800_000_000_000;

fn curate_engine_with(llm: Arc<dyn LlmProvider>, embed: Arc<dyn EmbedProvider>) -> Engine {
    Engine::new(
        Arc::new(SqliteStore::open_in_memory().unwrap()),
        llm,
        embed,
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(CURATE_NOW)),
    )
}

fn mem_id(e: &Engine, bank: &str, needle: &str) -> String {
    e.store
        .list_memories(bank, 1000)
        .unwrap()
        .into_iter()
        .find(|m| m.text.contains(needle))
        .unwrap_or_else(|| panic!("no memory containing {needle:?}"))
        .id
}

fn causal(bank: &str, a: &str, b: &str) -> Edge {
    Edge {
        bank_id: bank.into(),
        a_id: a.into(),
        b_id: b.into(),
        kind: EdgeKind::Causal,
        label: Some("causes".into()),
        weight: 1.0,
    }
}

fn has_causal(e: &Engine, bank: &str, a: &str, b: &str) -> bool {
    e.store
        .edges_for_bank(bank, 10_000)
        .unwrap()
        .iter()
        .any(|x| x.kind == EdgeKind::Causal && x.a_id == a && x.b_id == b)
}

/// The store's semantic search agrees with the exhaustive scan.
fn assert_semantic_matches_scan(e: &Engine, bank: &str, query: &[f32]) {
    let kinds = [
        MemoryKind::World,
        MemoryKind::Experience,
        MemoryKind::Observation,
        MemoryKind::Opinion,
    ];
    let all = e.store.all_embeddings(bank).unwrap();
    let scan = brute_force_semantic(&all, &[query], &kinds, 50, None);
    let got = e
        .store
        .search_semantic(bank, &[query], &kinds, 50, None)
        .unwrap();
    let ids = |h: &[SemanticHit]| h.iter().map(|x| x.0.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&got[0]), ids(&scan[0]));
}

async fn embed_text(text: &str) -> Vec<f32> {
    MockEmbed::default()
        .embed(vec![text.into()])
        .await
        .unwrap()
        .remove(0)
}

async fn recall_ids(e: &Engine, bank: &str, query: &str) -> Vec<String> {
    e.recall(RecallRequest {
        bank_id: bank.into(),
        query: query.into(),
        token_budget: 5000,
        top_k_per_arm: 50,
        kinds: None,
        ..Default::default()
    })
    .await
    .unwrap()
    .memories
    .into_iter()
    .map(|m| m.id)
    .collect()
}

#[tokio::test]
async fn curation_edit_reembeds_drops_observations_and_keeps_causal_edges() {
    let e = engine();
    retain(
        &e,
        "ed",
        "Alice joined Google. Alice works on search infrastructure.",
    )
    .await;
    let joined = mem_id(&e, "ed", "joined Google");
    let works = mem_id(&e, "ed", "search infrastructure");
    e.store.add_edges(&[causal("ed", &joined, &works)]).unwrap();
    e.consolidate_once("ed", 10).await.unwrap();
    assert_eq!(e.store.observations_for_bank("ed").unwrap().len(), 1);
    assert!(e.store.list_dirty_entities("ed", 10).unwrap().is_empty());
    let alice_entities = e.store.memory_entity_ids(&works).unwrap();
    assert!(!alice_entities.is_empty());

    let mut req = CurateRequest::new("ed", &works);
    req.edit.text = Some("Alice works on the ads ranking team".into());
    let unit = e.curate_memory(req).await.unwrap();
    assert_eq!(unit.text, "Alice works on the ads ranking team");
    assert_eq!(unit.edited_at_ms, Some(CURATE_NOW));
    assert_eq!(unit.state, MemoryState::Valid);
    // Entities untouched by a text-only edit.
    assert_eq!(e.store.memory_entity_ids(&works).unwrap(), alice_entities);

    // The observation citing the fact is gone; its causal edge survives;
    // its entities are queued again.
    assert!(e.store.observations_for_bank("ed").unwrap().is_empty());
    assert!(has_causal(&e, "ed", &joined, &works));
    let dirty = e.store.list_dirty_entities("ed", 10).unwrap();
    assert!(
        alice_entities.iter().all(|x| dirty.contains(x)),
        "{dirty:?}"
    );

    // Re-embedded and re-indexed: the new text is found, the old is not.
    let old_hits = e
        .store
        .search_bm25("ed", &["infrastructure".into()], 10)
        .unwrap();
    assert!(old_hits.iter().all(|h| h.0 != works), "{old_hits:?}");
    let new_hits = e.store.search_bm25("ed", &["ranking".into()], 10).unwrap();
    assert!(new_hits.iter().any(|h| h.0 == works));
    let q = embed_text("Alice works on the ads ranking team").await;
    let sem = e
        .store
        .search_semantic("ed", &[&q], &[MemoryKind::World], 1, None)
        .unwrap();
    assert_eq!(sem[0][0].0, works);
    assert!(sem[0][0].2 > 0.99);
    assert_semantic_matches_scan(&e, "ed", &q);
    assert!(recall_ids(&e, "ed", "ads ranking team")
        .await
        .contains(&works));

    // Consolidation rebuilds an observation from the edited text.
    e.consolidate_once("ed", 10).await.unwrap();
    let obs = e.store.observations_for_bank("ed").unwrap();
    assert!(
        obs.iter()
            .any(|o| o.text.contains("ads ranking")
                && o.evidence.iter().any(|ev| ev.memory_id == works)),
        "{obs:?}"
    );
}

#[tokio::test]
async fn curation_invalidate_hides_everywhere_and_restore_brings_it_back() {
    let e = engine();
    retain(
        &e,
        "inv",
        "Alice joined Google. Alice works on search infrastructure.",
    )
    .await;
    let joined = mem_id(&e, "inv", "joined Google");
    let works = mem_id(&e, "inv", "search infrastructure");
    e.store
        .add_edges(&[causal("inv", &joined, &works)])
        .unwrap();
    e.consolidate_once("inv", 10).await.unwrap();
    let entities_before = e.store.memory_entity_ids(&works).unwrap();
    assert!(recall_ids(&e, "inv", "search infrastructure")
        .await
        .contains(&works));

    let mut req = CurateRequest::new("inv", &works);
    req.state = Some(MemoryState::Invalidated);
    req.reason = Some("superseded".into());
    let unit = e.curate_memory(req).await.unwrap();
    assert_eq!(unit.state, MemoryState::Invalidated);
    assert_eq!(unit.invalidation_reason.as_deref(), Some("superseded"));
    assert_eq!(unit.invalidated_at_ms, Some(CURATE_NOW));

    // Gone from every recall arm.
    assert!(!recall_ids(&e, "inv", "search infrastructure")
        .await
        .contains(&works));
    assert!(e
        .store
        .search_bm25("inv", &["infrastructure".into()], 10)
        .unwrap()
        .iter()
        .all(|h| h.0 != works));
    let q = embed_text("Alice works on search infrastructure").await;
    assert!(e
        .store
        .search_semantic("inv", &[&q], &[MemoryKind::World], 10, None)
        .unwrap()[0]
        .iter()
        .all(|h| h.0 != works));
    assert_semantic_matches_scan(&e, "inv", &q);
    assert!(!e
        .store
        .neighbors("inv", std::slice::from_ref(&joined), 2, 100)
        .unwrap()
        .contains(&works));
    assert!(e
        .store
        .memories_in_window("inv", 0, i64::MAX, 100)
        .unwrap()
        .iter()
        .all(|m| m.id != works));
    // No link rows left behind; the citing observation is deleted.
    assert!(e.store.memory_entity_ids(&works).unwrap().is_empty());
    assert!(e
        .store
        .edges_for_bank("inv", 10_000)
        .unwrap()
        .iter()
        .all(|x| x.a_id != works && x.b_id != works));
    assert!(e.store.observations_for_bank("inv").unwrap().is_empty());
    let archived = e
        .list_memory_units(
            "inv",
            &MemoryListQuery {
                state: Some(MemoryState::Invalidated),
                limit: 100,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(archived.1, 1);
    assert_eq!(archived.0[0].id, works);

    // A reason-only update of an archived row.
    let mut req = CurateRequest::new("inv", &works);
    req.state = Some(MemoryState::Invalidated);
    req.reason = Some("server decommissioned".into());
    let unit = e.curate_memory(req).await.unwrap();
    assert_eq!(
        unit.invalidation_reason.as_deref(),
        Some("server decommissioned")
    );

    // Restore: recall, entity links and the causal edge come back.
    let mut req = CurateRequest::new("inv", &works);
    req.state = Some(MemoryState::Valid);
    let unit = e.curate_memory(req).await.unwrap();
    assert_eq!(unit.state, MemoryState::Valid);
    assert!(unit.invalidation_reason.is_none());
    assert!(recall_ids(&e, "inv", "search infrastructure")
        .await
        .contains(&works));
    assert_eq!(e.store.memory_entity_ids(&works).unwrap(), entities_before);
    assert!(has_causal(&e, "inv", &joined, &works));
    assert_semantic_matches_scan(&e, "inv", &q);
    assert!(e
        .list_memory_units(
            "inv",
            &MemoryListQuery {
                state: Some(MemoryState::Invalidated),
                limit: 100,
                ..Default::default()
            },
        )
        .unwrap()
        .0
        .is_empty());
}

#[tokio::test]
async fn curation_invalidated_fact_is_invisible_to_reflect_recall() {
    let script = MockLlm::script(vec![
        causal_facts_resp(&[
            ("Gus maintains the billing service.", &[]),
            ("Gus maintains the payroll service.", &[]),
        ]),
        tool_call("1", "recall", json!({"query": "Gus maintains service"})),
    ]);
    let e = curate_engine_with(Arc::new(script), Arc::new(MockEmbed::default()));
    retain(&e, "rf", "Gus maintains billing and payroll.").await;
    let billing = mem_id(&e, "rf", "billing");
    let payroll = mem_id(&e, "rf", "payroll");
    let mut req = CurateRequest::new("rf", &billing);
    req.state = Some(MemoryState::Invalidated);
    e.curate_memory(req).await.unwrap();

    // The finish cites both facts; only what the recall tool retrieved
    // survives citation validation.
    let finish = tool_call(
        "2",
        "finish",
        json!({"answer": "Gus maintains services.", "citations": [billing, payroll]}),
    );
    let e2_script = MockLlm::script(vec![
        tool_call("1", "recall", json!({"query": "Gus maintains service"})),
        finish.clone(),
        finish,
    ]);
    let e = Engine::new(
        Arc::clone(&e.store),
        Arc::new(e2_script),
        Arc::new(MockEmbed::default()),
        Arc::new(MockRerank),
        Arc::new(SimpleTokenEstimator),
        Arc::new(FixedClock::new(CURATE_NOW)),
    );
    let resp = e.reflect("rf", "What does Gus maintain?").await.unwrap();
    assert!(resp.citations.contains(&payroll), "{resp:?}");
    assert!(!resp.citations.contains(&billing), "{resp:?}");
}

#[tokio::test]
async fn curation_resolve_entities_literal_vs_fuzzy() {
    let e = engine();
    retain(&e, "ent", "Someone joined the search team.").await;
    let fact = mem_id(&e, "ent", "search team");
    let existing = e
        .store
        .create_entity(&NewEntity {
            bank_id: "ent".into(),
            name: "Jonathan Smith".into(),
            etype: EntityType::Person,
        })
        .unwrap();

    // Literal: a near-miss name becomes its own entity.
    let mut req = CurateRequest::new("ent", &fact);
    req.edit.entities = Some(vec!["Jonathon Smith".into()]);
    req.resolve_entities = false;
    let unit = e.curate_memory(req).await.unwrap();
    assert_eq!(unit.entities, vec!["Jonathon Smith".to_string()]);
    let linked = e.store.memory_entity_ids(&fact).unwrap();
    assert_eq!(linked.len(), 1);
    assert_ne!(linked[0], existing.id);

    // Literal still reuses a case-insensitive exact match.
    let mut req = CurateRequest::new("ent", &fact);
    req.edit.entities = Some(vec!["jonathan smith".into()]);
    req.resolve_entities = false;
    e.curate_memory(req).await.unwrap();
    assert_eq!(
        e.store.memory_entity_ids(&fact).unwrap(),
        vec![existing.id.clone()]
    );

    // Fuzzy (retain's resolver): the near-miss lands on the existing one.
    let mut req = CurateRequest::new("ent", &fact);
    req.edit.entities = Some(vec!["Jonathan Smyth".into()]);
    let unit = e.curate_memory(req).await.unwrap();
    assert_eq!(e.store.memory_entity_ids(&fact).unwrap(), vec![existing.id]);
    assert_eq!(unit.entities, vec!["Jonathan Smith".to_string()]);

    // [] detaches every entity.
    let mut req = CurateRequest::new("ent", &fact);
    req.edit.entities = Some(vec![]);
    let unit = e.curate_memory(req).await.unwrap();
    assert!(unit.entities.is_empty());
    assert!(e.store.memory_entity_ids(&fact).unwrap().is_empty());
}

fn archived(e: &Engine, bank: &str) -> usize {
    e.list_memory_units(
        bank,
        &MemoryListQuery {
            state: Some(MemoryState::Invalidated),
            limit: 100,
            ..Default::default()
        },
    )
    .unwrap()
    .1
}

async fn invalidate(e: &Engine, bank: &str, id: &str) {
    let mut req = CurateRequest::new(bank, id);
    req.state = Some(MemoryState::Invalidated);
    e.curate_memory(req).await.unwrap();
}

#[tokio::test]
async fn curation_archive_follows_document_and_bank_deletion() {
    let e = engine();
    let retain_doc = |doc: &'static str, content: &'static str| {
        e.retain(RetainRequest {
            bank_id: "arc".into(),
            content: content.into(),
            document_id: Some(doc.into()),
            ..Default::default()
        })
    };
    retain_doc("d1", "Hana likes tea. Hana owns a bicycle.")
        .await
        .unwrap();
    invalidate(&e, "arc", &mem_id(&e, "arc", "tea")).await;
    assert_eq!(archived(&e, "arc"), 1);
    e.forget_document("arc", "d1").unwrap();
    assert_eq!(archived(&e, "arc"), 0);

    // update_mode: replace drops the old version's archived rows too.
    retain_doc("d2", "Ivan likes coffee. Ivan owns a kayak.")
        .await
        .unwrap();
    invalidate(&e, "arc", &mem_id(&e, "arc", "coffee")).await;
    assert_eq!(archived(&e, "arc"), 1);
    retain_doc("d2", "Ivan likes cocoa now.").await.unwrap();
    assert_eq!(archived(&e, "arc"), 0);

    retain_doc("d3", "Jun likes juice.").await.unwrap();
    invalidate(&e, "arc", &mem_id(&e, "arc", "juice")).await;
    assert_eq!(archived(&e, "arc"), 1);
    e.delete_bank("arc").await.unwrap();
    assert_eq!(archived(&e, "arc"), 0);
}

#[tokio::test]
async fn curation_refuses_with_upstream_errors() {
    let e = engine();
    retain(
        &e,
        "ref",
        "Alice joined Google. Alice works on search infrastructure.",
    )
    .await;
    e.consolidate_once("ref", 10).await.unwrap();
    let fact = mem_id(&e, "ref", "joined Google");
    let obs = e.store.observations_for_bank("ref").unwrap()[0].id.clone();

    let mut req = CurateRequest::new("ref", &obs);
    req.state = Some(MemoryState::Invalidated);
    let err = e.curate_memory(req).await.unwrap_err();
    assert!(
        matches!(&err, Error::BadInput(m) if m.contains("observation")),
        "{err:?}"
    );

    let err = e
        .curate_memory(CurateRequest::new("ref", &fact))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::BadInput(m) if m.contains("at least one")),
        "{err:?}"
    );

    let mut req = CurateRequest::new("ref", "not-a-uuid");
    req.state = Some(MemoryState::Invalidated);
    let err = e.curate_memory(req).await.unwrap_err();
    assert!(
        matches!(&err, Error::BadInput(m) if m.contains("UUID")),
        "{err:?}"
    );

    let mut req = CurateRequest::new("ref", uuid_like());
    req.state = Some(MemoryState::Invalidated);
    let err = e.curate_memory(req).await.unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");

    invalidate(&e, "ref", &fact).await;
    let mut req = CurateRequest::new("ref", &fact);
    req.edit.text = Some("Alice joined Alphabet".into());
    let err = e.curate_memory(req).await.unwrap_err();
    assert!(
        matches!(&err, Error::BadInput(m) if m.contains("revert")),
        "{err:?}"
    );

    let mut req = CurateRequest::new("ref", &fact);
    req.edit.kind = Some(MemoryKind::Observation);
    let err = e.curate_memory(req).await.unwrap_err();
    assert!(
        matches!(&err, Error::BadInput(m) if m.contains("fact_type")),
        "{err:?}"
    );
}

fn uuid_like() -> String {
    "00000000-0000-4000-8000-000000000000".into()
}

#[tokio::test]
async fn curation_failed_embed_leaves_the_row_untouched() {
    let embed = Arc::new(SwitchEmbed {
        inner: MockEmbed::default(),
        fail: false.into(),
    });
    let e = curate_engine_with(Arc::new(MockLlm::demo()), embed.clone());
    retain(&e, "fe", "Kim plays the cello.").await;
    let fact = mem_id(&e, "fe", "cello");
    let before = e.get_memory_unit("fe", &fact).unwrap().unwrap();
    let entities = e.store.memory_entity_ids(&fact).unwrap();

    embed.fail.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut req = CurateRequest::new("fe", &fact);
    req.edit.text = Some("Kim plays the viola".into());
    req.edit.entities = Some(vec!["Viola Society".into()]);
    assert!(e.curate_memory(req).await.is_err());
    let after = e.get_memory_unit("fe", &fact).unwrap().unwrap();
    assert_eq!(after.text, before.text);
    assert_eq!(after.edited_at_ms, None);
    assert_eq!(e.store.memory_entity_ids(&fact).unwrap(), entities);
    assert!(e
        .store
        .entities_for_bank("fe")
        .unwrap()
        .iter()
        .all(|x| x.name != "Viola Society"));

    // Invalidate needs no provider; restore does, and must not half-apply.
    invalidate(&e, "fe", &fact).await;
    let mut req = CurateRequest::new("fe", &fact);
    req.state = Some(MemoryState::Valid);
    assert!(e.curate_memory(req).await.is_err());
    let unit = e.get_memory_unit("fe", &fact).unwrap().unwrap();
    assert_eq!(unit.state, MemoryState::Invalidated);
    assert!(e.store.get_memory(&fact).unwrap().is_none());
}
