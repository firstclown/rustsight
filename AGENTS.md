# AGENTS.md — rustsight

A Hindsight-style agent memory engine in Rust, built as an embeddable component
for a desktop harness. From-scratch Rust reimplementation of the architecture in
"Hindsight is 20/20" (arXiv:2512.12818), with prompts informed by the MIT-licensed
`vectorize-io/hindsight` reference (pinned tag `v1.1.0`). Not a code port — see
`NOTICE` for attribution rules.

## Commands

> `cargo` lives in `~/.cargo/bin` and may not be on a non-interactive shell's
> PATH. If `cargo: command not found`, run:
> `export PATH="$HOME/.cargo/bin:$PATH"`

```bash
cargo build                              # debug build (workspace)
scripts/build-release                    # release binary WITH the local ONNX
                                         # reranker (CUDA; --cpu for CPU-only)
                                         # — parity/benchmark runs need this.
                                         # A plain `cargo build --release`
                                         # overwrites the binary WITHOUT the
                                         # reranker feature and it refuses to
                                         # start under RUSTSIGHT_RERANK=local
cargo test                               # all tests; fully offline, mock providers, no API key needed
cargo test -p rustsight-storage --test engine_flow   # just the end-to-end integration suite
cargo test -p rustsight-core --test upstream_prompt_parity   # upstream prompt parity suite (35 cases)
cargo fmt --check                        # formatting check (must pass; no rustfmt.toml, defaults only)
cargo fmt                                # auto-fix formatting
cargo clippy --workspace --all-targets -- -D warnings # lints (must pass with zero warnings)
```

Prompt vendor tooling (RS-5):

```bash
scripts/prompt-sync      # re-vendor upstream prompt sources at the pinned tag; diffs vs previous vendor
scripts/prompt-goldens   # regenerate src/prompts/upstream/texts.rs + parity fixtures (python3 + pydantic)
```

Run the CLI from source:

```bash
cargo run -p rustsight-cli -- --mock --db /tmp/demo.db retain --bank demo \
    --content "Alice joined Google. Alice works on search infrastructure."
cargo run -p rustsight-cli -- --mock --db /tmp/demo.db consolidate --bank demo
cargo run -p rustsight-cli -- --mock --db /tmp/demo.db recall --bank demo --query "Where does Alice work?"
cargo run -p rustsight-cli -- --mock --db /tmp/demo.db reflect --bank demo --query "What should I know about Alice?"
cargo run -p rustsight-cli -- serve --addr 127.0.0.1:8888   # HTTP API (needs API key or --mock);
                                                            # also accepts hindsight-api's --host/--port/--log-level
cargo run -p rustsight-cli -- mcp --bank demo                # stdio MCP server
```

Smoke-test the Hindsight-compatible API (what Hindsight clients call):

```bash
curl -X PUT localhost:8888/v1/default/banks/demo -H 'content-type: application/json' \
    -d '{"name": "Demo", "reflect_mission": "Remember things."}'
curl -X POST localhost:8888/v1/default/banks/demo/memories -H 'content-type: application/json' \
    -d '{"items": [{"content": "Alice joined Google.", "document_id": "d1", "tags": ["t"]}]}'
curl -X POST localhost:8888/v1/default/banks/demo/memories/recall -H 'content-type: application/json' \
    -d '{"query": "Where does Alice work?", "tags": ["t"], "tags_match": "any_strict"}'
```

Smoke-test the native HTTP API by hand:

```bash
curl -X POST localhost:8888/banks/demo/retain -H 'content-type: application/json' \
    -d '{"content": "Alice joined Google."}'
curl -X POST localhost:8888/banks/demo/recall -H 'content-type: application/json' \
    -d '{"query": "Where does Alice work?"}'
```

## Definition of Done

Every change is complete only when **all** of the following pass locally:

1. `cargo fmt --check` — zero diffs.
2. `cargo clippy --workspace --all-targets -- -D warnings` — zero warnings.
3. `cargo test` — all tests green. New behavior ships with tests:
   - Unit tests live inline in the module under `mod tests` (see
     `crates/rustsight-core/src/fusion.rs`, `temporal.rs`, `prompts/*.rs`).
   - Engine-level/integration tests go in
     `crates/rustsight-storage/tests/engine_flow.rs` (in-memory SQLite +
     scripted mocks; 69 tests there today).
4. Docs touched where relevant: `docs/reference.md` (and the README overview) for user-facing behavior/env
   vars, `NEXT_STEPS.md` checkboxes if a ticket's scope changed.
5. No secrets/API keys committed; `.gitignore` already covers `*.db*`.

There is no CI yet (ticketed as RS-12 in `NEXT_STEPS.md`) — the checks above are
the gate, so run them before declaring done.

## Workspace layout

Five crates; dependency flow: `core` ← `storage`/`providers` ← `server` ←
`cli`.

| Crate | Path | Role |
|---|---|---|
| rustsight-core | `crates/rustsight-core` | Types, `MemoryStore` trait, provider traits, prompts, engine pipelines (retain/recall/reflect/consolidate/mental) |
| rustsight-storage | `crates/rustsight-storage` | `SqliteStore`: single SQLite file, WAL, FTS5 BM25, sqlite-vec KNN index over BLOB embeddings (`vector.rs`, brute-force fallback), typed edge graph |
| rustsight-providers | `crates/rustsight-providers` | `OpenAiCompat` LLM/embeddings over HTTP, `ApiRerank` (Jina/Cohere-style `/rerank`), `NoopRerank`, and all mocks (`MockLlm`, `MockEmbed`, `MockRerank`, `SimpleTokenEstimator`) |
| rustsight-server | `crates/rustsight-server` | axum HTTP router (`http.rs`) + stdio JSON-RPC MCP server (`mcp.rs`); both spawn the consolidation worker and the webhook delivery worker (`webhooks.rs`) |
| rustsight-cli | `crates/rustsight-cli` | Binary `rustsight`: serve/mcp/retain/recall/reflect/consolidate/inspect/banks/bank-set/mental-model/reindex |

Key source map:

- `crates/rustsight-core/src/engine/mod.rs` — `Engine` + `EngineConfig`; engine
  is constructor-injected with `Arc<dyn ...>` for store, LLM, embed, rerank,
  token estimator, and clock. `spawn_worker(interval, max_entities)` runs
  background consolidation.
- `crates/rustsight-core/src/engine/{retain,recall,reflect,consolidate,mental}.rs`
  — the five pipelines. Retain chunks content (`engine/chunks.rs`: token
  split with overlap, causal-index remap, overlap-dedup at similarity >= 0.9,
  half-size re-ask on parse failure) and retries transient provider errors
  (`with_retry` in `engine/mod.rs` — HTTP 429/5xx typed `Error::LlmStatus`/
  `EmbedStatus`, plus transport-level drops/timeouts as `Error::LlmTransport`/
  `EmbedTransport`). Consolidation sends the byte-parity upstream prompt per
  dirty entity and applies the creates/updates/deletes plan with a bank-level
  exact-dupe guard (`engine/consolidate.rs`, RS-17). Reflect runs a
  kind-separated tool loop (`search_observations`/`recall`/`read_observation`/
  `finish`, RS-18: query-decomposition guidance, #4567 sufficiency gate,
  upstream tool budget defaults) — then the closing ask (upstream
  `_ask_for_done`: a forced `finish` over the full conversation) writes
  the answer; the loop's volunteered finish is the fallback and a separate
  final-synthesis call runs only as the last resort (`engine/reflect.rs`;
  "reflect produced no answer" needs a failed synthesis with no loop answer).
- `crates/rustsight-core/src/store.rs` — the `MemoryStore` trait (swap-in point
  for Postgres/ANN backends).
- `crates/rustsight-core/src/traits.rs` — `LlmProvider`, `EmbedProvider`,
  `RerankProvider`, `TokenEstimator`, `Clock` (+ `SystemClock`/`FixedClock`).
- `crates/rustsight-core/src/prompts/` — extraction/consolidation/reflect prompt
  builders. Extraction, consolidation, and the reflect answer machine use the
  vendored upstream prompts (RS-5/RS-17, pinned tag `v1.1.0`):
  `src/prompts/upstream/` holds the byte-parity ports and
  `crates/rustsight-core/prompts/upstream/` the vendored sources + pin +
  DIVERGENCES.md (which records the declared engine divergences — per-entity
  consolidation batches, the reflect tool loop's own tool set, hand-ported
  final-synthesis builders pending golden fixtures).
- `crates/rustsight-storage/src/sqlite.rs` — schema + all queries (~1.3k lines;
  schema inline at top).
- `crates/rustsight-server/src/compat.rs` — the Hindsight wire-compatible
  HTTP API (RS-9): `/v1/default/banks/...`, `/version`, `/health`, in
  hindsight-api 1.2.x request/response shapes, merged into the same router
  as the native API. Shapes were checked against the installed upstream
  `api/http.py` and exercised with the real `@vectorize-io/hindsight-client`;
  `docs/reference.md`'s compatibility matrix lists what is accepted-but-ignored.
- `crates/rustsight-core/src/bank_config.rs` — the Hindsight-named bank
  settings (`CONFIG_KEYS`, validation, `Engine::update_bank_config`) and
  `Engine::provision_bank`; `templates.rs` — bank template manifests
  (parse/import/export) and `Engine::reflect_directives` (RS-10d).
- `crates/rustsight-core/src/files.rs` — `FileParser` trait, `ParserRegistry`
  (built-in `text`/`html`), content-addressed `Attachment` (RS-10c);
  `crates/rustsight-server/src/files.rs` serves `files/retain` and
  `attachments/{id}`.
- `crates/rustsight-cli/src/main.rs` — `build_engine` is the three-line
  embedding recipe desktop apps copy.

## Architecture invariants (do not break)

- **Never hold the SQLite connection lock (`Mutex<Connection>` in `SqliteStore`)
  across an LLM call.** The store is sync; take the lock only for short queries
  between async provider calls, or retain/recall latency blocks.
- **Consolidation is single-flight.** `Engine::consolidate_once` serializes via
  `consolidate_lock`; each dirty entity's row is cleared only after *that
  entity's* observations are written and embedded (per-entity progress — a
  pass that errors, stalls, or hits its soft deadline keeps every finished
  entity's work; `consolidate_timeout_secs` stops starting new entities and
  the hard wall at 2x only cuts wedged in-flight calls). Keep those
  semantics.
- **Retain/recall response paths never wait on consolidation.** Retain enqueues
  dirty entities and notifies the worker (`wakeup: Notify`); the worker
  debounces bursts, then consolidates every bank from `banks_with_dirty()`.
- **Reflect citations are validated against retrieved memory ids** — unknown ids
  are filtered out (see `engine/reflect.rs`). Keep that filter.
- **The reflect answer comes from the closing ask** (upstream
  `agent.py::_ask_for_done`: after the loop stops — volunteered `finish`,
  exhaustion, or silence — one closing user turn asks for the `finish`
  with the tool forced; that answer wins). The loop's volunteered finish
  is the fallback; the separate final-synthesis call over the packed tool
  history runs only when neither produced anything. `ReflectResponse.answer_path`
  records loop | forced-done | synthesis. A premature `finish` with
  `answer_found: false` before any `recall` is refused once (#4567 gate).
  Harness-noise content (`<|channel|>`-style tokens) is treated as an
  empty turn, and every answer is stripped of inline memory ids (harvested
  into citations). The loop's `recall` tool is kind-filtered
  (world/experience); observations come only via `search_observations`.
  Opinion extraction runs in a background task after the response is
  composed — never on its critical path.
- **Opinions link to canonical entity ids** (`entity_ids`) so consolidation can
  reinforce them; the linkage flows through entity resolution
  (`entity.rs` aliases), not free-text matching.
- Recall = four parallel arms (vector cosine, FTS5 BM25, 2-hop graph, parsed
  temporal window) → RRF fuse (`fusion.rs`) → rerank (whole fused pool;
  `reranker_max_candidates` 0 = uncapped) → combined scoring (sigmoid/passthrough
  CE normalization × recency/proof-count boosts, `engine/scoring.rs`) → greedy
  token-budget cut. Reflect's pool width comes from `recall_budget`
  (fixed MID = 300/arm, upstream `_resolve_thinking_budget`); graph seeds = 20
  (`GRAPH_SEED_LIMIT`). RRF `k=60` and other knobs
  live in `EngineConfig`.
- **Documents own their memories.** Every retain files its memories under
  a document (given or generated). `update_mode: replace` deletes the old
  version's memories — FTS rows, entity links, edges, and observation
  evidence that cited them — only *after* extraction and embedding
  succeeded; `append` keeps them. Entities whose facts changed are
  re-queued for consolidation. Keep the "provider calls first, destructive
  store writes last" order.
- **Tag scopes filter inside each recall arm, before its top-k cut**
  (`TagFilter` in `types.rs`, upstream `tags_match` semantics). Observation
  tags are derived (union of evidence facts' tags via `tag_index`), never
  stored.
- **A bank configured through the Hindsight API scopes `mission` to
  reflect** (`bank_config.mission_scope = "reflect"`); retain and
  consolidation read `retain_mission` / `observations_mission` through
  `Engine::scoped_mission`. Natively configured banks keep sharing
  `mission`.
- **The native reflect path is unchanged by request budgets.**
  `Engine::reflect` runs `reflect_max_rounds` (8); only requests that name
  a budget (the compat API, default `low`) use upstream's
  `reflect_budget_base_rounds` (10) x 0.5/1/2.
- **Bank deletion goes through `Engine::delete_bank`.** Anything writing to
  a bank across an await holds a `BankGuard` (`engine/bank_gate.rs`) and runs
  under `until_deleted`, so a delete cancels it and cannot interleave with
  it. Never take a second guard for the same bank in one task: a queued
  delete would deadlock it; pass the guard (`Engine::retain_guarded`).
- **Memory Defense screens before anything is derived from content**
  (RS-10a, `defense.rs`, `retain.rs::screen_content`): the bank's
  `memory_defense` policy runs on `req.content` before chunking, so a
  `redact` reaches the document body, chunks, facts and embeddings alike.
  A malformed policy fails the retain — never skip screening (fail closed).
  The pattern catalog is held to upstream by `memory_defense_parity.rs`;
  regenerate with `scripts/defense-goldens`, never hand-edit the fixture.
- **Events are emitted after the work they report is durable**
  (`Engine::emit`, `events.rs`, RS-10b): `retain.completed` after the
  memories are stored, `consolidation.completed` once per pass that found
  work, `memory_defense.triggered` for every redact/block. Sinks run
  synchronously, so they must only queue (the webhook sink writes delivery
  rows) — never make network calls inside `EventSink::emit`.
- **Banks are created through `Engine::provision_bank`** (RS-10d): it is
  what applies the default bank template, exactly once, before any caller
  settings. Don't create banks with `store.ensure_bank` in new code paths.
- **`enable_observations: false` gates consolidation, not retain.** Dirty
  entities keep queueing so re-enabling catches up.
- **Embeddings are mirrored write-through into the vec0 index** (RS-7,
  `crates/rustsight-storage/src/vector.rs`): any store method that writes,
  updates, or deletes a memory/observation/opinion embedding must call
  `vec_put`/`vec_remove` in the same transaction. A bank answers from the
  index only while `vec_indexed_banks` lists it; a store opened without the
  index clears every marker, and `reindex_vectors` rebuilds. Every
  `search_semantic` implementation must return what `brute_force_semantic`
  returns (`assert_matches_scan` in the storage tests).
- Schema changes are additive column migrations in `SqliteStore::migrate`
  (`CREATE TABLE IF NOT EXISTS` never alters an existing table) — add new
  columns there, with a test that opens an old-schema file.
- Timestamps are ms-epoch i64 everywhere; embeddings are LE f32 BLOBs.

## Testing patterns

- Deterministic by construction: `SqliteStore::open_in_memory()`,
  `MockLlm::script(vec![...])` for scripted LLM turns (extraction JSON → tool
  calls → opinion JSON), `MockLlm::demo()` for loose end-to-end flows,
  `FixedClock::new(ms)` for time control. No network, no env vars, no sleeps
  beyond bounded poll loops (see `background_worker_consolidates_after_retain_without_explicit_call`
  for the pattern, including the long-interval trick to prove the notify path).
- Worker tests must `worker.abort(); let _ = worker.await;` at the end.
- Scripted-LLM order matters: retain extraction first, then reflect's
  search/recall/finish tool calls, then the closing-ask turn (always sent:
  script a `finish` tool call to make it answer, or plain prose to make it
  yield to the loop's volunteered answer — no synthesis turn unless both
  loop-side answers are missing), then opinion extraction (backgrounded —
  poll the store before making further scripted calls), then
  consolidation's observation synthesis and reinforcement verdicts. Read an
  existing test in `engine_flow.rs` before writing a new one.

## Configuration (non-mock mode)

Env vars, read in `crates/rustsight-providers/src/env.rs`:

- `RUSTSIGHT_LLM_BASE_URL` (default `https://api.openai.com/v1`),
  `RUSTSIGHT_LLM_API_KEY` (required unless `--mock`), `RUSTSIGHT_LLM_MODEL`
  (default `gpt-4o-mini`)
- `RUSTSIGHT_RETAIN_LLM_*` / `RUSTSIGHT_REFLECT_LLM_*` /
  `RUSTSIGHT_CONSOLIDATION_LLM_*` (`_BASE_URL`/`_API_KEY`/`_MODEL`; optional
  per-operation models, unset parts inherit the main LLM). Every LLM
  variable also answers to its `HINDSIGHT_API_*` spelling (rustsight name
  wins) so a Hindsight supervisor's environment works unchanged.
- `RUSTSIGHT_EMBED_BASE_URL`/`RUSTSIGHT_EMBED_API_KEY`/`RUSTSIGHT_EMBED_MODEL`
  (default to the LLM values / `text-embedding-3-small`)
- `RUSTSIGHT_HTTP_TIMEOUT_SECS` (default 300, clamped 5–3600; total wall for
  one provider HTTP round trip, connect through last body byte; invalid
  values fall back to default with a warning)
- `RUSTSIGHT_RETAIN_MAX_COMPLETION_TOKENS` (or its `HINDSIGHT_API_*` spelling;
  default 64000; output cap per fact-extraction call, lowered per model in
  `OpenAiCompat`; invalid values fall back to default with a warning)
- `RUSTSIGHT_RERANK_URL` optional (Jina/Cohere-style `/rerank`); unset →
  `NoopRerank` (raw RRF ordering — accuracy caveat, RS-4 tracks a local ONNX
  reranker)
- `RUSTSIGHT_DB` (default `~/.rustsight/rustsight.db`)
- `RUSTSIGHT_VECTOR_INDEX` (default on; `off` keeps semantic search on the
  brute-force scan and invalidates the index markers — run
  `rustsight reindex` after turning it back on)
- `RUSTSIGHT_CONSOLIDATE_INTERVAL_MS` (default 5000; worker cadence, invalid
  values fall back to default with a warning)
- `RUSTSIGHT_ASYNC_RETAIN_CONCURRENCY` (default 2, clamped 1–64; async
  retains running at once through the compat API, invalid values fall back
  to default with a warning)
- `RUSTSIGHT_DEFAULT_BANK_TEMPLATE` (inline JSON or a file path; RS-10d,
  read by `rustsight_server::default_bank_template_from_env`, installed with
  `Engine::with_default_template`).
- `RUSTSIGHT_ENABLE_FILE_UPLOAD_API`, `RUSTSIGHT_FILE_PARSER` (default
  `text,html`), `RUSTSIGHT_FILE_CONVERSION_MAX_BATCH_SIZE[_MB]` (RS-10c;
  read in `crates/rustsight-server/src/files.rs`).
- `RUSTSIGHT_WEBHOOK_URL` / `_SECRET` / `_EVENT_TYPES`,
  `RUSTSIGHT_WEBHOOK_ALLOWED_HOSTS`, `RUSTSIGHT_WEBHOOK_EXPOSE_RESPONSE_BODY`
  (RS-10b; read in `crates/rustsight-server/src/webhooks.rs`, each also
  answers to `HINDSIGHT_API_*`).
- `RUSTSIGHT_API_TOKEN` / `RUSTSIGHT_API_TOKEN_FILE` (optional; bearer-token
  auth for the HTTP API — without one, `serve` refuses non-loopback binds
  unless `--allow-unauth`. MCP stdio needs no auth.)

## Conventions

- Edition 2021, deps declared once in the root `[workspace.dependencies]` and
  referenced with `workspace = true`. Add new deps there, not per-crate.
- Default rustfmt; format on save or run `cargo fmt` before committing.
- Commit style: short imperative subject referencing the ticket, e.g.
  `RS-2: add background consolidation worker`. Never commit without an explicit
  request.
- `NEXT_STEPS.md` is the roadmap (RS-1..RS-12, P0..P3, each with need-to-know
  facts and its own DoD). RS-1..RS-5 are done; RS-6's in-repo eval harness was
  removed (benchmarks now run externally through the Agent Memory Benchmark). Read the relevant ticket
  before touching engine code — it records upstream-pinned facts (tag `v1.1.0`,
  commit `a5a619b5`) you'd otherwise have to rediscover.
- License is MIT; upstream material keeps MIT headers and a `NOTICE` entry.
  Update `NOTICE` when vendoring anything new.
