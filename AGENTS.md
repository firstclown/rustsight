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
cargo build --release                    # release build (thin LTO); binary at target/release/rustsight
cargo test                               # all 85 tests; fully offline, mock providers, no API key needed
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
cargo run -p rustsight-cli -- serve --addr 127.0.0.1:8888   # HTTP API (needs API key or --mock)
cargo run -p rustsight-cli -- mcp --bank demo                # stdio MCP server
```

Smoke-test HTTP by hand:

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
     scripted mocks; 16 tests there today).
4. Docs touched where relevant: README sections for user-facing behavior/env
   vars, `NEXT_STEPS.md` checkboxes if a ticket's scope changed.
5. No secrets/API keys committed; `.gitignore` already covers `*.db*`.

There is no CI yet (ticketed as RS-12 in `NEXT_STEPS.md`) — the checks above are
the gate, so run them before declaring done.

## Workspace layout

Five crates; dependency flow: `core` ← `storage`/`providers` ← `server` ← `cli`.

| Crate | Path | Role |
|---|---|---|
| rustsight-core | `crates/rustsight-core` | Types, `MemoryStore` trait, provider traits, prompts, engine pipelines (retain/recall/reflect/consolidate/mental) |
| rustsight-storage | `crates/rustsight-storage` | `SqliteStore`: single SQLite file, WAL, FTS5 BM25, brute-force cosine over BLOB embeddings, typed edge graph |
| rustsight-providers | `crates/rustsight-providers` | `OpenAiCompat` LLM/embeddings over HTTP, `ApiRerank` (Jina/Cohere-style `/rerank`), `NoopRerank`, and all mocks (`MockLlm`, `MockEmbed`, `MockRerank`, `SimpleTokenEstimator`) |
| rustsight-server | `crates/rustsight-server` | axum HTTP router (`http.rs`) + stdio JSON-RPC MCP server (`mcp.rs`); both spawn the consolidation worker |
| rustsight-cli | `crates/rustsight-cli` | Binary `rustsight`: serve/mcp/retain/recall/reflect/consolidate/inspect/banks/bank-set/mental-model |

Key source map:

- `crates/rustsight-core/src/engine/mod.rs` — `Engine` + `EngineConfig`; engine
  is constructor-injected with `Arc<dyn ...>` for store, LLM, embed, rerank,
  token estimator, and clock. `spawn_worker(interval, max_entities)` runs
  background consolidation.
- `crates/rustsight-core/src/engine/{retain,recall,reflect,consolidate,mental}.rs`
  — the five pipelines. Retain chunks content (`engine/chunks.rs`: token
  split with overlap, causal-index remap, overlap-dedup at similarity >= 0.9,
  half-size re-ask on parse failure) and retries transient provider errors
  (`with_retry` in `engine/mod.rs`, 429/5xx only — typed `Error::LlmStatus`/
  `EmbedStatus`).
- `crates/rustsight-core/src/store.rs` — the `MemoryStore` trait (swap-in point
  for Postgres/ANN backends).
- `crates/rustsight-core/src/traits.rs` — `LlmProvider`, `EmbedProvider`,
  `RerankProvider`, `TokenEstimator`, `Clock` (+ `SystemClock`/`FixedClock`).
- `crates/rustsight-core/src/prompts/` — extraction/consolidation/reflect prompt
  builders. Extraction prompts are the vendored upstream ones (RS-5, pinned
  tag `v1.1.0`): `src/prompts/upstream/` holds the byte-parity ports and
  `crates/rustsight-core/prompts/upstream/` the vendored sources + pin +
  DIVERGENCES.md. Consolidation/reflect prompts are still informed rewrites
  until their response machines are ported (see DIVERGENCES.md).
- `crates/rustsight-storage/src/sqlite.rs` — schema + all queries (~1.3k lines;
  schema inline at top).
- `crates/rustsight-cli/src/main.rs` — `build_engine` is the three-line
  embedding recipe desktop apps copy.

## Architecture invariants (do not break)

- **Never hold the SQLite connection lock (`Mutex<Connection>` in `SqliteStore`)
  across an LLM call.** The store is sync; take the lock only for short queries
  between async provider calls, or retain/recall latency blocks.
- **Consolidation is single-flight.** `Engine::consolidate_once` serializes via
  `consolidate_lock`; dirty-entity rows are only cleared *after* a successful
  pass so crashes are safe (work is re-discovered next tick). Keep that
  ordering.
- **Retain/recall response paths never wait on consolidation.** Retain enqueues
  dirty entities and notifies the worker (`wakeup: Notify`); the worker
  debounces bursts, then consolidates every bank from `banks_with_dirty()`.
- **Reflect citations are validated against retrieved memory ids** — unknown ids
  are filtered out (see `engine/reflect.rs`). Keep that filter.
- **Opinions link to canonical entity ids** (`entity_ids`) so consolidation can
  reinforce them; the linkage flows through entity resolution
  (`entity.rs` aliases), not free-text matching.
- Recall = four parallel arms (vector cosine, FTS5 BM25, 2-hop graph, parsed
  temporal window) → RRF fuse (`fusion.rs`) → rerank (capped at
  `min(50, refs.len())`) → greedy token-budget cut. RRF `k=60` and other knobs
  live in `EngineConfig`.
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
  recall/finish tool calls, then opinion extraction, then consolidation's
  observation synthesis and reinforcement verdicts. Read an existing test in
  `engine_flow.rs` before writing a new one.

## Configuration (non-mock mode)

Env vars, read in `crates/rustsight-providers/src/env.rs`:

- `RUSTSIGHT_LLM_BASE_URL` (default `https://api.openai.com/v1`),
  `RUSTSIGHT_LLM_API_KEY` (required unless `--mock`), `RUSTSIGHT_LLM_MODEL`
  (default `gpt-4o-mini`)
- `RUSTSIGHT_EMBED_BASE_URL`/`RUSTSIGHT_EMBED_API_KEY`/`RUSTSIGHT_EMBED_MODEL`
  (default to the LLM values / `text-embedding-3-small`)
- `RUSTSIGHT_RERANK_URL` optional (Jina/Cohere-style `/rerank`); unset →
  `NoopRerank` (raw RRF ordering — accuracy caveat, RS-4 tracks a local ONNX
  reranker)
- `RUSTSIGHT_DB` (default `~/.rustsight/rustsight.db`)
- `RUSTSIGHT_CONSOLIDATE_INTERVAL_MS` (default 5000; worker cadence, invalid
  values fall back to default with a warning)
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
  facts and its own DoD). RS-1 and RS-2 are done. Read the relevant ticket
  before touching engine code — it records upstream-pinned facts (tag `v1.1.0`,
  commit `a5a619b5`) you'd otherwise have to rediscover.
- License is MIT; upstream material keeps MIT headers and a `NOTICE` entry.
  Update `NOTICE` when vendoring anything new.
