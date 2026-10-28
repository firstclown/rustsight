# rustsight — Next Steps

Tickets for the gap between the current commit and full Hindsight parity, in
recommended execution order. Priorities: **P0** correctness gaps in shipped
code, **P1** accuracy-defining work, **P2** scale/surface/features, **P3** the
end goal. Sizes: **S** < a day, **M** a few days, **L** a week or more.

Context for all tickets: upstream reference is vectorize-io/hindsight
(MIT, pinned tag `v1.1.0`, release commit `a5a619b5`, Oct 21 2026); paper is
arXiv:2512.12818. Engine code lives in `crates/rustsight-core/src/engine/`,
storage in `crates/rustsight-storage/src/sqlite.rs`.

Suggested sequence: RS-1 → RS-2 → RS-3 → RS-13 (CI) → RS-4 → RS-5 → RS-6 →
RS-11, with RS-7…RS-10 interleaved as needed.

---

## RS-1 — Activate opinion reinforcement (dead code path) · P0 · S

### Description
The CARA belief-evolution loop is implemented but unreachable. Opinions are
created with `entity_ids: vec![]` in `form_opinions()`
(`crates/rustsight-core/src/engine/reflect.rs`), while the consolidation-time
reinforcement filter in `crates/rustsight-core/src/engine/consolidate.rs`
selects opinions via `o.entity_ids.iter().any(|id| id == eid)`. Empty
`entity_ids` means no opinion is ever selected for reinforcement, so confidence
scores and history never evolve after formation.

### Need-to-know facts
- Storage already supports everything needed: the `opinion_entities` table
  exists and `upsert_opinion()` in `sqlite.rs` links rows from `o.entity_ids`.
- Entity resolution machinery exists (`entity.rs`,
  `store.find_by_alias`/`entities_for_bank`); opinions just never go through it.
- Two viable linking strategies: (a) resolve entities named in the opinion text
  against bank aliases at formation time; (b) have the opinion-extraction LLM
  output entity mentions in its JSON (`{"opinions":[{"text","confidence",
  "entities":[...]}]}`), mirroring the fact-extraction schema. (b) matches
  upstream's approach of LLM-tagged metadata and reuses `ExtractedEntity`
  parsing.
- Reinforcement correctness depends on `Opinion.updated_at_ms` vs fact
  `mentioned_at_ms` (already implemented in `consolidate.rs`).

### Definition of done
- [x] New opinions are linked to canonical entities at formation time.
- [x] Integration test: form an opinion about entity E, retain new facts about
      E, run `consolidate_once`, assert the opinion's confidence moved and a
      `ConfidencePoint` was appended to history (scripted LLM returning
      `{"changed":true,...}`).
- [x] Unit test: opinion with no matching entities is untouched by
      consolidation.
- [x] `reflect` response or inspect output exposes linked entities for
      opinions.

---

## RS-2 — Background consolidation worker · P0 · S

### Description
Consolidation only runs when explicitly invoked (`rustsight consolidate`,
`POST /banks/{id}/consolidate`). Upstream runs a continuous worker off the
dirty-entity queue so observations appear without anyone asking. rustsight
needs: a worker task spawned alongside `serve`, plus an automatic,
non-blocking trigger after `retain`.

### Need-to-know facts
- The queue already exists: `mark_entities_dirty` / `list_dirty_entities` /
  `clear_dirty_entities` in `sqlite.rs`; `Engine::consolidate_once` drains it.
- `consolidate_once` makes LLM calls that can take seconds — it must never
  block the retain/recall response path. Retain should enqueue and return;
  the worker picks the queue up.
- Needs a single-flight guard (an `AtomicBool` or `tokio::sync::Mutex` on the
  engine) so overlapping runs don't double-process; SQLite `dirty_entities`
  rows are only deleted at the end of a successful pass, so crashes are safe
  (work is re-discovered).
- The store is sync behind a `Mutex<Connection>`; consolidation holds the lock
  only for short queries between LLM calls — keep it that way (no LLM call
  while holding the connection lock).
- Library embedders (Tauri) need this too: expose
  `Engine::spawn_worker(interval)` rather than baking it into `serve()` only.

### Definition of done
- [x] `Engine::spawn_worker(interval, max_entities)` returns a join handle and
      loops: sleep → `consolidate_once` for every bank with a non-empty dirty
      queue (needs a `banks_with_dirty` store method or iterate `list_banks`).
- [x] `serve` (HTTP) and the MCP server spawn the worker; graceful shutdown on
      ctrl-c cancels it.
- [x] Retain triggers a debounce-flavored immediate wakeup (channel or
      `tokio::select!` on notify) without blocking its own response.
- [x] Test: retain → await worker tick (or notify) → observation exists
      without any explicit consolidate call.
- [x] Config: `RUSTSIGHT_CONSOLIDATE_INTERVAL_MS` (default ~5s) documented in
      README.

---

## RS-3 — Document chunking and provider retries · P0 · M

### Description
`retain` sends the entire content blob to the LLM in one request. Long
transcripts will exceed context or produce truncated extraction output, and
any transient provider failure fails the whole retain. Upstream chunks
documents, auto-recurses when output is too long, and retries
(`extract_facts_from_contents`, `build_chunk_prompt_parts`, auto-split on
`OutputTooLongError` in `hindsight-api-slim/.../retain/fact_extraction.py`).

### Need-to-know facts
- Chunking must be token-based; the `TokenEstimator` trait is the measuring
  stick (see RS-12 for real tokenizer). Sane starting point: ~3k-token chunks
  with ~200-token overlap, both configurable in `EngineConfig`.
- **Causal index remapping**: extraction returns `causal[].fact_index` local
  to one chunk's fact list. When merging chunks, offsets must be remapped to a
  global index before `retain.rs` builds edges, or edges will point at the
  wrong facts.
- Overlap will produce duplicate facts across chunk boundaries; merge by
  `string_similarity >= 0.9` before insertion.
- Retries: exponential backoff (2–3 attempts) on `Error::Llm` for 429/5xx only;
  the OpenAI-compat provider surfaces status in the message — plumb a typed
  status through instead of string matching.
- JSON-mode truncation detection: if `parse_extraction` fails on a chunk, one
  re-ask at half the chunk size before giving up (upstream's auto-split).

### Definition of done
- [x] `EngineConfig` gains chunk size/overlap; retain splits, extracts per
      chunk, remaps causal indexes, dedupes overlap facts.
- [x] Retry-with-backoff on transient LLM/embedding failures.
- [x] Re-ask at half size on parse failure (one recursion level).
- [x] Tests: two-chunk content with a cross-chunk causal link produces the
      correct edge; duplicate-from-overlap is stored once; a scripted 429 then
      success path succeeds.
- [x] README documents limits.

---

## RS-4 — Real reranking: local ONNX cross-encoder + always-on default · P1 · M

### Description
`NoopRerank` ships as the default (unset `RUSTSIGHT_RERANK_URL`), so recall
order is raw RRF. The paper attributes a significant share of benchmark lead
to cross-encoder reranking. Need a local ONNX reranker (bge-reranker-v2-m3 via
the `ort` crate) behind a cargo feature, and make reranking on-by-default for
real (non-mock) deployments.

### Need-to-know facts
- Trait seam exists: `RerankProvider` in `core/src/traits.rs`; HTTP API
  reranker already implemented (`ApiRerank`, Jina/Cohere-shaped `/rerank`).
- `ort` is a heavy dependency (ONNX runtime ~50–100 MB) — must be
  opt-in via feature `onnx-rerank`, not a default dep; build a binary variant
  with the feature for releases.
- Model acquisition: bge-reranker-v2-m3 (~2 GB fp32; int8 quant ~300 MB) from
  HuggingFace. Need first-run download with a cache dir
  (`~/.rustsight/models/`), plus an offline path (pre-placed file +
  `RUSTSIGHT_RERANK_MODEL_PATH`).
- Cross-encoder batching: score query+doc pairs in batches (16–64) —
  naive per-doc calls are 10–50x slower.
- Recall currently truncates rerank input to `min(50, refs.len())`
  (`engine/recall.rs`); keep that cap.
- Upstream also has a "decision-model reranker" with pruning
  (`search/decision_reranker.py`) — out of scope here, note for later.

### Definition of done
- [x] `rustsight-providers` feature `onnx-rerank` implementing `OrtRerank`
      with batching, model download/cache, env config
      (`RUSTSIGHT_RERANK=local`).
- [x] Default provider selection: real API key + no rerank config → warn
      loudly or default to API rerank if key/URL present; document the
      accuracy implication.
- [x] Fixture test: given 5 docs with known relevance order, reranked recall
      puts the relevant doc first (NoopRerank does not).
- [x] Latency note in README: p50/p95 added to recall at 100 docs.
- [x] Mock mode untouched (MockRerank stays for offline demo/tests).

Done Oct 2026, plus a CUDA extension beyond the original ticket
(`onnx-rerank-cuda` feature, `RUSTSIGHT_RERANK_DEVICE`). Facts for later
tickets:

- Default model repos are device-paired: CPU → `rafa754/bge-reranker-v2-m3`
  int8 (`onnx/model_quantized.onnx`, ~544 MB, ~3x faster on CPU than the
  onnx-community int8 export); CUDA →
  `onnx-community/bge-reranker-v2-m3-ONNX` fp16 (`onnx/model_fp16.onnx`,
  ~1.1 GB, the only fp16 export found). Measured on a 16-core laptop CPU +
  RTX 3050 Ti: CPU int8 p50 1.78s / CUDA fp16 p50 0.29s at 50 docs.
- **int8-on-CUDA trap**: int8 exports use `MatMulInteger`, which has no CUDA
  kernels — ORT runs those ops on CPU with device copies and the result is
  *slower* than pure CPU (2.8s vs 1.8s at 50 docs). Never register the CUDA
  EP for the int8 file; `cuda_paired_repo(_file)` in
  `rustsight-providers/src/onnx.rs` enforces the pairing.
- ONNX Runtime **silently** falls back to CPU when CUDA provider libs are
  missing (registration never errors), so `auto` probes the stack by
  dlopen (driver + `libcudart.so.13`/`libcublas{,Lt}.so.13`/
  `libcurand.so.10`) before promising CUDA.
- fp32 exports keep weights in an external `model.onnx_data` blob; the
  resolver downloads it alongside when present (HEAD probe).
- Upstream's "decision-model reranker" with pruning (`search/
  decision_reranker.py`) remains out of scope; if built, it layers on the
  same `RerankProvider` seam.

---

## RS-5 — Vendor upstream prompts with sync script and golden parity tests · P1 · M

### Description
Current prompts (`core/src/prompts/`) are informed rewrites, not the
production prompts that give upstream its accuracy. Vendor the real prompt
source from a pinned tag, port the builder functions faithfully, and prove
rendering parity with fixtures derived from upstream's own prompt tests.

### Need-to-know facts
- Source files (at tag `v1.1.0`, path prefix `hindsight-api-slim/hindsight_api/engine/`):
  - `retain/fact_extraction.py` (~3.8k lines; prompt builder
    `_build_extraction_prompt_and_schema`, config shapes: default /
    string-pattern / entity-labels / both)
  - `consolidation/prompts.py` (266 lines; `build_consolidation_system_prompt`,
    standard/custom modes, mission section)
  - `reflect/prompts.py` (1.76k lines) and `reflect/tools.py`/`tools_schema.py`
  - `search/temporal_extraction.py` (147 lines, query-side time parsing)
- Upstream test files usable as golden fixtures:
  `test_fact_extraction_fact_type_prompt.py`, `test_reflect_prompt_builder.py`,
  `test_prompt_brace_escape.py`, `test_prompt_context_truncation.py`,
  `test_llm_judge_prompt.py`, `test_delta_prompt_schema_parity.py`.
- The prompts are **builder functions**, not static strings — config
  (labels, patterns, output language, mission, disposition) composes into the
  final text. Python f-string/`.format` translation to Rust `format!` is
  mechanical except brace-escaping (`{{`/`}}`) — upstream has a dedicated test
  for it.
- License: repo-root MIT (Copyright 2025 Vectorize AI, Inc.) — the
  Apache-2.0 line in `hindsight-api/README.md` is stale; root LICENSE governs.
  Keep MIT headers on vendored files + NOTICE entry (NOTICE already
  attributes; extend with the pinned commit).
- Upstream exposes `POST /banks/{id}/prompts/preview` for rendered prompts —
  consider a `rustsight prompt-preview` CLI subcommand as the debugging
  equivalent.

### Definition of done
- [x] `scripts/prompt-sync` (bash) fetches the four upstream files at a pinned
      tag/commit, writes them under
      `crates/rustsight-core/prompts/upstream/<tag>/`, and diffs against the
      previous vendor to surface upstream prompt changes.
- [x] `PROMPTS_VERSION` file records tag + commit hash; NOTICE updated.
- [x] Rust builders ported so that, given identical config inputs, output
      matches the ported upstream fixtures byte-for-byte (`insta` snapshot
      tests, at least: extraction default + labels + custom-instruction +
      language-directive shapes; consolidation standard/custom; reflect
      system with/without mission).
- [x] Deliberate divergences (e.g., our reflect tool names) listed in
      `prompts/upstream/DIVERGENCES.md`.
- [x] `cargo test` runs the parity suite; sync script documented in README.

Done Oct 2026. Facts for later tickets:

- Parity oracle is upstream's own Python: `scripts/prompt-goldens` assembles
  a stub harness around the vendored files, runs the real builders, and
  writes `tests/fixtures/upstream_prompts/*.txt` + regenerates
  `src/prompts/upstream/texts.rs` (all big prompt text comes from this
  generator — nothing hand-transcribed). 35 cases incl. the full reflect
  tools-prompt branch matrix; volatile clock frozen to
  `2026-08-09 14:32 UTC` exactly like upstream's own test.
- Tests assert BOTH `assert_eq` against the fixture and an insta snapshot
  (`.snap` bodies are upstream bytes, proven equal at generation time).
  After a sync+goldens round, refresh snapshots deliberately:
  `INSTA_FORCE_UPDATE=1 cargo test -p rustsight-core`, then review.
- Escape semantics to remember when porting more builders: values
  substituted into a `.format()` template keep their braces **doubled**
  (extraction custom instructions, `build_final_system_prompt` mission);
  text embedded **into** a template that is then formatted as a whole comes
  out single (consolidation mission/capacity). Both pinned by fixtures.
- Production retain now sends the upstream prompts; the parser accepts both
  upstream and legacy response shapes (see `parse_extraction`), so scripted
  tests and older providers keep working. Consolidation/reflect pipelines
  keep their own prompts until their response machines are ported — see
  `prompts/upstream/DIVERGENCES.md` for exactly what adopting requires.

---

## RS-6 — LongMemEval-S evaluation harness · P1 · L

### Description
No evidence yet that the Rust engine approaches the paper's accuracy. Build
the `rustsight-eval` crate (it was in the original workspace plan but was
deferred) that runs LongMemEval-S end-to-end: retain each session's
transcript, answer the 500 questions via `reflect`, score with an LLM judge.

### Need-to-know facts
- LongMemEval-S: ~115k tokens, ~50 sessions per conversation, 500 questions
  across five abilities — information extraction, multi-session reasoning,
  temporal reasoning, knowledge update, abstention. Dataset is on HuggingFace
  (`xiamen-unity/LongMemEval` for the original; verify the exact variant the
  paper used before running).
- Paper reference points (Table 3): Hindsight OSS-20B **83.6%**, OSS-120B
  **89.0%**, Gemini-3-Pro answer head **91.4%**; full-context OSS-20B
  baseline 39.0%. Our first target: beat full-context baseline with a 20B-class
  extraction+reflect model, then iterate.
- Judge setup per paper: GPT-OSS-120b, temperature 0, binary correctness,
  judge prompt from paper Appendix A.4; abstention scored separately.
- Cost/time: retain processes ~115k tokens per conversation × the dataset's
  conversations; extraction dominates. Make bank-per-conversation, resumable
  (persist banks, skip answered questions), and parallel over conversations.
- The harness must use the same provider stack as production (OpenAI-compat),
  with Ollama as the local option. Record model names, costs, wall time.
- LoCoMo (second paper benchmark, 50 conversations, multimodal) is a stretch
  goal behind a flag — text-only first.

### Definition of done
- [ ] `rustsight-eval` crate with `cargo run -p rustsight-eval -- --dataset
      longmemeval-s --out results/` that ingests, retains, answers, judges.
- [ ] Per-category and overall accuracy report (markdown + JSON), including
      abstention handling; resumable runs; per-question traces (retrieved
      ids, citations, answer) written for failure analysis.
- [ ] One completed baseline run recorded in `evals/RESULTS.md` with
      model/config/cost, compared against the paper's table.
- [ ] At least one iteration informed by the failure traces (prompt or fusion
      tweak) with measured delta.

---

## RS-7 — ANN vector backend (sqlite-vec) · P2 · M

### Description
Semantic recall scans every embedding row and computes cosine in Rust
(`all_embeddings` in `sqlite.rs`). Correct but O(n) per query; fine to ~50k
memories, sluggish beyond. Add a `sqlite-vec` backed index behind the same
`MemoryStore` surface.

### Need-to-know facts
- `sqlite-vec` ships as a loadable extension with a bundled-build cargo
  crate; verify it loads alongside rusqlite's bundled SQLite (version
  compatibility) before committing to it — fallback options: `usearch`
  in-process, or a persisted HNSW sidecar.
- Embeddings are stored as LE f32 BLOBs — same encoding works for
  sqlite-vec's vec0 virtual table.
- Observations and opinions carry embeddings too (three tables today);
  either unify vectors into one vec table keyed by (kind, id) or maintain
  three indexes — one unified table is simpler to keep consistent.
- Reindex path needed for existing DBs (read all BLOBs → insert into vec
  table); keep brute-force as the fallback when the extension fails to load.

### Definition of done
- [ ] Feature `sqlite-vec` in `rustsight-storage`; vec0 table + triggers or
      write-through inserts; `search_semantic` uses KNN when enabled.
- [ ] Migration/reindex command (`rustsight reindex --bank`).
- [ ] Criterion bench: recall semantic arm at 10k / 100k vectors, brute force
      vs ANN, recall@10 quality delta reported.
- [ ] Graceful fallback when extension unavailable (log + brute force).
- [ ] README scale guidance updated.

---

## RS-8 — Knowledge pages and `fs mount` · P2 · M

### Description
Mental models exist (`engine/mental.rs`, `mental_models` table with a `path`
column already reserved) but there is no page tree, no refresh cascade, and no
filesystem projection. Implement knowledge pages as mental models organized
in folders, rewritten incrementally as consolidation produces new knowledge,
plus a `rustsight fs-mount` command that materializes the tree as markdown.

### Need-to-know facts
- Upstream semantics: pages answer one question each, live in a folder tree,
  are built from observations (not raw transcript), **never read other
  pages** (prevents citation feedback loops), and refresh in the background
  when consolidation changes their scope.
- Refresh cascade: after `consolidate_once` touches entity E, pages whose
  question/scope mentions E are stale — simplest v1: mark all pages stale on
  any consolidation and refresh one per worker tick (bounded LLM spend).
- `fs mount` upstream is a real projection (FUSE-optional); v1 here should be
  a sync command writing a directory of markdown files (front-matter with
  id/question/last_refreshed), leaving FUSE as a later ticket.
- Page content must cite observations by id in front-matter or footnotes for
  auditability.

### Definition of done
- [ ] Pages addressable by path; CLI: `page create --bank --path --question`,
      `page list`, `page refresh`.
- [ ] Worker refreshes stale pages incrementally (one per tick).
- [ ] `rustsight fs-mount --bank --dir` writes the tree; re-running is
      idempotent (overwrite in place); deleted pages removed.
- [ ] Test: create page, retain + consolidate new facts, worker tick, page
      content changes; page never contains another page's content.
- [ ] README section with example tree.

---

## RS-9 — Hindsight-client API compatibility layer · P2 · M

### Description
The HTTP API is custom-shaped. Validate a compatibility subset against
upstream's published OpenAPI spec so existing Hindsight clients (their
Python/TS/Go SDKs, coding-agent integrations) can point at a rustsight server
for basic retain/recall/reflect.

### Need-to-know facts
- Upstream generates clients from an OpenAPI spec shipped in the repo
  (`hindsight-api`, spec regenerated per release — pin to v1.1.0's spec).
- Field shapes differ today (e.g., our `RetainResponse {memories_stored,
  entities, edges_created}` vs their retain result); timestamps are ms ints
  here vs ISO strings there. Compatibility = accept *their* request bodies
  and return *their* response shapes on a compat router — do not break the
  native API.
- Bank-scoped MCP upstream is `POST /mcp/{bank_id}/` (streamable HTTP); ours
  is stdio-only — an HTTP MCP endpoint belongs in this ticket too if cheap.
- Aim for "inspired-by compatible", documented subset; upstream evolves fast
  (3,900+ commits), don't promise full parity.

### Definition of done
- [ ] `/v1/compat` router (feature-gated) implementing retain/recall/reflect
      + bank CRUD per upstream v1.1.0 spec subset, with their field names.
- [ ] OpenAPI document generated for our native API (utoipa or hand-written
      spec) and served at `/openapi.json`.
- [ ] Smoke test: `pip install hindsight-client` pointed at rustsight
      performs retain → recall → reflect successfully (scripted in
      `rustsight-system-tests` or CI job).
- [ ] Compat matrix doc: supported endpoints, known divergences.

---

## RS-10 — Upstream feature catch-up (individually ticketed) · P2 · varies

### Description
Features present upstream that rustsight deliberately lacks. Each should be
pulled out into its own ticket when scheduled; listed here with enough context
to size them.

### Need-to-know facts / sub-items
- **Memory defense (S/M):** opt-in per-bank policy scanning every retain for
  secrets/PII (upstream ships 45 patterns) with `redact` or `block` actions
  (e.g. `[REDACTED:github_token]`). Implement as a retain pre-parse hook;
  pattern set is a plain data file upstream — port it.
- **Documents & attachments (M):** retain accepts documents with attachments
  (files stored via `attachment_store`, parsed by pluggable parsers —
  markitdown/llama_parse/iris upstream). v1: file ingest with plain-text
  extraction + attachment ids on facts.
- **Async operations + webhooks (M):** long retains return an operation id;
  status polling + webhook delivery on retain/consolidation/refresh
  lifecycle events. Needs an `operations` table and a small event emitter.
- **Bank templates (S):** declarative bank presets (mission/directives/
  disposition/extraction labels) — upstream `engine/bank_templates.py`;
  mostly config plumbing on our `Bank`.
- **Multilingual preservation (S/M):** input-language detection and
  preservation end-to-end (entities keep native script, facts untranslated);
  FTS5 tokenizer choice matters for CJK (upstream ships
  `chinese_temporal_periods.py` and trigram indexes — note our FTS uses
  `porter unicode61`, which tokenizes CJK poorly; trigram or per-language
  tokenizer needed).
- **Per-tenant/hierarchical config (M):** global env → per-tenant → per-bank
  override chain upstream; ours is env-only today.

### Definition of done
- [ ] Each sub-item broken into its own ticket with tests before work starts;
      this umbrella closes when triaged.

---

## RS-11 — The desktop harness (the actual goal) · P3 → should become P0 once RS-1..6 land · L

### Description
rustsight exists to be the memory component of a new Rust desktop harness.
That harness hasn't been started. Build the host application: a desktop shell
that embeds the engine in-process, exposes MCP tools to its agents, and gives
a UI over banks/memories/observations.

### Need-to-know facts
- Embedding path: `rustsight-core` + `rustsight-storage` compile into a
  Tauri backend directly (rusqlite bundled works fine inside Tauri on
  Linux/macOS/Windows). Alternative for process isolation: ship the
  `rustsight` binary as a Tauri sidecar and talk HTTP.
- Engine construction is three lines (see `rustsight-cli/src/main.rs
  build_engine`); mock mode gives a fully offline demo path for development.
- The HTTP inspect endpoints (`/banks/{id}/memories|entities|observations|
  opinions`) already return everything a memory-inspector UI needs.
- Worker from RS-2 must be spawned in the Tauri setup hook; single SQLite
  file at `~/.rustsight/rustsight.db` (or app-data dir) is the only state.
- MCP: harness agents get tools by spawning `rustsight mcp` (stdio) or
  hitting the HTTP server; either already works.

### Definition of done
- [ ] Tauri (or equivalent) app scaffolded in this repo or a sibling repo,
      embedding the engine in-process.
- [ ] UI: bank switcher, retain box, recall search with kinds/strategies
      shown, memory/observation/opinion inspector, mental-model viewer.
- [ ] Agent loop wired to MCP tools (retain-before/reflect-after pattern like
      upstream's litellm wrapper).
- [ ] Packaging: dev build runs with `--mock` providers offline; release
      build with real provider config via settings UI.
- [ ] README at repo root links the harness.

---

## RS-12 — Tokenizer, benches, CI, release hygiene · P2 · S each

### Description
Small quality items that keep getting deferred; do them opportunistically.

### Need-to-know facts / sub-items
- **tiktoken-rs (S):** implement `TokenEstimator` with real BPE behind a
  feature; keep `SimpleTokenEstimator` for tests/mocks. Budget accuracy
  currently ±30% on code-heavy text (chars/4).
- **Criterion benches (S):** `rustsight-storage/benches`: BM25 query, cosine
  scan at 1k/10k/100k, graph BFS, RRF at 500 candidates, end-to-end recall.
  Needed as the before/after baseline for RS-7.
- **CI (S):** GitHub Actions: `cargo fmt --check`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo test` on
  ubuntu/macOS/windows; release build artifact upload on tags.
- **Publish prep (S):** crate metadata/descriptions (mostly present), `cargo
  package` dry-run for `rustsight-core`, doc examples on `Engine` methods.

### Definition of done
- [ ] Each sub-item landed with its named artifact green.

---

## Appendix — facts referenced by multiple tickets

- Upstream pin: `vectorize-io/hindsight` tag `v1.1.0`, commit `a5a619b5`,
  MIT (root LICENSE, Copyright 2025 Vectorize AI, Inc.). Engine source path:
  `hindsight-api-slim/hindsight_api/engine/`.
- Paper: arXiv:2512.12818, TEMPR (retain/recall) + CARA (reflect); four
  networks W/B/O/S; RRF + cross-encoder + token budget; LongMemEval-S
  reference scores 83.6 / 89.0 / 91.4 (OSS-20B / OSS-120B / Gemini-3 head).
- Current test baseline: 39 tests green, clippy clean (as of initial commit
  `9070afb`).
- Deviations already accepted: brute-force vectors (RS-7), informed-rewrite
  prompts (RS-5), sync-only consolidation (RS-2), custom HTTP shapes (RS-9).
