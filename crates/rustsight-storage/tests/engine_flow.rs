use rustsight_core::Engine;
use rustsight_core::*;
use rustsight_providers::{
    FlakyEmbed, FlakyLlm, MockEmbed, MockLlm, MockRerank, SimpleTokenEstimator,
};
use rustsight_storage::SqliteStore;
use serde_json::json;
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
    retain(&e, "r", "Alice works at Google as an engineer.").await;
    let resp = e.reflect("r", "What does Alice do?").await.unwrap();
    assert_eq!(resp.answer, "Alice works at Google.");
    assert!(resp.rounds >= 2);
    assert!(resp.retrieved >= 1);
    assert!(
        resp.citations.is_empty(),
        "bogus citation must be filtered: {resp:?}"
    );
    assert_eq!(resp.opinions_formed, 0);
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
    assert_eq!(resp.opinions_formed, 1);
    let ops = e.store.list_opinions("o").unwrap();
    assert_eq!(ops.len(), 1);
    assert!((ops[0].confidence - 0.8).abs() < 1e-6);
    assert!(ops[0].embedding.is_some());
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
    assert_eq!(resp.opinions_formed, 1);

    let alice = e
        .store
        .entities_for_bank("ri")
        .unwrap()
        .into_iter()
        .find(|x| x.name == "Alice")
        .expect("alice entity exists");
    let ops = e.store.list_opinions("ri").unwrap();
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
        })
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::LlmStatus { status: 400, .. }),
        "expected typed 400, got: {err:?}"
    );
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
