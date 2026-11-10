use async_trait::async_trait;
use rustsight_core::Engine;
use rustsight_core::*;
use rustsight_providers::{
    FlakyEmbed, FlakyLlm, MockEmbed, MockLlm, MockRerank, NoopRerank, SimpleTokenEstimator,
};
use rustsight_storage::SqliteStore;
use serde_json::json;
use std::collections::{HashMap, VecDeque};
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
        LlmResponse {
            content: Some(r#"{"opinions":[]}"#.into()),
            tool_calls: vec![],
        },
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

    let script = MockLlm::script(vec![
        LlmResponse {
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
        },
        LlmResponse {
            content: Some(r#"{"opinions":[]}"#.into()),
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
    let resp = e.reflect("batch", "What does Alice do?").await.unwrap();
    assert_eq!(resp.answer, "Alice works at Google.");
    assert_eq!(resp.rounds, 1, "batched recall+finish is one round");
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
    let resp = e.reflect("quiet", "Anybody there?").await.unwrap();
    assert_eq!(resp.answer, "Recovered after a nudge.");
    assert_eq!(resp.rounds, 2);
}

/// Two consecutive empty responses still fail — the nudge is a retry, not
/// an infinite loop; rounds and the streak bound it.
#[tokio::test]
async fn reflect_fails_after_consecutive_empty_responses() {
    let store = SqliteStore::open_in_memory().unwrap();
    let script = MockLlm::script(vec![
        LlmResponse {
            content: None,
            tool_calls: vec![],
        },
        LlmResponse {
            content: None,
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
    let err = e.reflect("silent", "q").await.unwrap_err();
    assert!(
        err.to_string().contains("empty response"),
        "unexpected error: {err}"
    );
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
        LlmResponse {
            content: Some(r#"{"opinions":[]}"#.into()),
            tool_calls: vec![],
        },
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
        LlmResponse {
            content: Some(r#"{"opinions":[]}"#.into()),
            tool_calls: vec![],
        },
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
        if system.contains("OBJECTIVE OBSERVER") {
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
        if system.contains("OBJECTIVE OBSERVER") {
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
    })
    .await
    .unwrap();
    clock.set(t0 + 3_600_000);
    e.retain(RetainRequest {
        bank_id: "dl".into(),
        content: "Bob bought a vintage synthesizer. Bob practices the synthesizer daily.".into(),
        context: None,
        timestamp_ms: Some(t0 + 3_600_000),
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
