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
  resolver downloads it alongside when present (HEAD probe). Cache dirs are
  stamped with a `.complete` marker after all files land, so cached models
  start with zero network calls (offline hosts work; pre-marker caches are
  adopted when the HEAD probe fails but the files are there). Downloads go
  through unique `<pid>.<seq>.part` temps and verify `Content-Length` plus
  the optional `RUSTSIGHT_RERANK_MODEL_SHA256` pin before renaming into
  place.
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

> **Status (2026-11-20): withdrawn.** The `rustsight-eval` crate, its
> fixtures, and `scripts/fullstack-smoke` / `scripts/hindsight-record-sync`
> were removed from the workspace — the harness never produced a full run.
> Benchmarks now run externally through the Agent Memory Benchmark's
> Hindsight HTTP adapter (results in the README). The notes below are kept
> as history; paths they mention no longer exist.

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
- [x] `rustsight-eval` crate with `cargo run -p rustsight-eval -- --dataset
      longmemeval-s --out results/` that ingests, retains, answers, judges.
- [x] Per-category and overall accuracy report (markdown + JSON), including
      abstention handling; resumable runs; per-question traces (retrieved
      ids, citations, answer) written for failure analysis.
- [ ] One completed baseline run recorded in `evals/RESULTS.md` with
      model/config/cost, compared against the paper's table.
- [ ] At least one iteration informed by the failure traces (prompt or fusion
      tweak) with measured delta.

Harness done Oct 2026 (remaining two boxes are the hosted baseline run +
iteration, which need API keys/money, not code). Facts for the run:

- Dataset is `longmemeval_s.json` from HF `xiaowu0162/longmemeval` — the
  ticket's `xiamen-unity/LongMemEval` is a deprecated mirror (decision:
  original variant only; `longmemeval-cleaned` deliberately not supported).
  The remote file is extensionless (`longmemeval_s`, 265 MB) and embeds raw
  newlines inside JSON strings, so ingest goes through a streaming sanitizer
  (`dataset::SanitizingReader`); 1,230 zero-turn placeholder sessions are
  dropped at ingest. Sessions carry no timestamps; the harness synthesizes
  them (1-day spacing default) ending at the latest `question_date` and
  passes them as retain `timestamp_ms`. Questions sharing a haystack share
  one bank (`lmes-<hash>`), so ~50 conversations are retained once each.
- Judge prompts follow paper App. A.4 / the LongMemEval convention:
  binary `\boxed{yes}`/`\boxed{no}` at temperature 0, abstention scored
  via an unanswerable-question variant; judge env is dedicated
  (`RUSTSIGHT_EVAL_JUDGE_BASE_URL/_API_KEY/_MODEL`, no silent fallback to
  the extraction model). Core `ReflectResponse` gained `retrieved_ids`
  (all memory ids surfaced across reflect rounds) to satisfy the trace DoD.
- The engine clock in eval runs is a `FixedClock` set to each question's
  `question_date`, anchoring temporal parsing; reflect queries carry a
  date prefix for the query-generation LLM.
- Consolidation runs via explicit `consolidate_once` loops until the dirty
  queue drains (bounded by pass limit + no-progress detection), not the
  background worker — deterministic and resumable. A bank whose
  consolidation stalls is kept *partially consolidated* (still answerable,
  `partial: true` in state.json) instead of failing the conversation. Cost
  model + launch procedure live in `evals/RESULTS.md` (~$1–4 for the full
  run via OpenRouter).

Postmortem (2026-10-31, first hosted attempt): the run retained 79/500
banks over 21h and completed zero. Three compounding flaws, all fixed:

1. `consolidate_once` was all-or-nothing — one malformed synthesis
   response, exhausted retry, or the 600s wall aborted the pass and threw
   away every entity's work; the eval loop treated that as a dead
   conversation. Now each entity is isolated: its observations are written,
   embedded, and its dirty row cleared per entity; failures leave only that
   entity dirty for the next pass (`engine/consolidate.rs`).
2. Passes ran per-entity LLM calls sequentially under a hard timeout, so
   slow batches predictably crossed the wall. `consolidate_timeout_secs` is
   now a soft deadline that stops *starting* entities (hard wall at 2x for
   wedged calls only), and `consolidate_concurrency` (default 4) fans
   synthesis out like retain's chunk extraction.
3. The runner deferred all outcome logging until every bank had been
   spawned, so 21h of failures printed nothing. Outcomes (and errors) now
   print the moment a conversation finishes, plus a `--heartbeat-secs`
   counter line.

Procedure change: before any full LongMemEval attempt, run
`scripts/fullstack-smoke` — a committed one-bank fixture (12 sessions, 7
questions across all five abilities) through the real provider stack
front-to-back (~minutes, ~cents). Green smoke first, then the hosted run.

Smoke postmortem (2026-11-04, first real-provider run, gpt-oss-20b via
OpenRouter): 5/7 correct, 0 partial — but three model-shape failures, all
fixed same day:

1. Consolidation parse: the model returned bare strings inside
   `observations` (not objects), failing the pass with "invalid type:
   string, expected struct RawObservation"; the dirty-row retry drained it
   on pass 3 anyway. `parse_observations` now takes strings as
   observations with default evidence (engine backfills from the first
   fact) — no pass wasted.
2. Reflect dropped batched tool calls: gpt-oss emits `recall` + `finish`
   in one turn; the first-call-only handler dropped the `finish` and the
   question died with "reflect produced no answer". The loop now executes
   every call in a turn (finish wins, same-batch recall results citable).
3. Reflect empty responses: gateway-stripped reasoning sometimes comes
   back as neither content nor tool calls ("reflect loop got an empty
   response"). The loop now nudges once with a user message (whitespace
   content counts as empty too); two consecutive empties still fail. The
   eval runner additionally retries a failed reflect once per question —
   in that smoke, 2/7 questions were permanent errors from flaws 2–3.

Parity reference (added Nov 2026): the per-question record of upstream
Hindsight's own LongMemEval-S run is committed at
`crates/rustsight-eval/fixtures/hindsight_longmemeval_s.json` (Agent Memory
Benchmark output: 473/500 = 94.6%, answer head `gemini-3.1-pro-preview`,
judge `gemini-2.5-flash-lite`, mode `rag`; query ids are the dataset's
`question_id`s, verified 500/500 against AMB's ingested split). After any
run, `cargo run -p rustsight-eval -- compare --run <dir>` writes
`compare.md`/`compare.json`: agreement matrix, per-ability deltas, parity
gaps (Hindsight right where rustsight is not — the iteration targets),
rustsight-only wins, shared failures. Hindsight's known weak spot in that
record: single-session-preference 23/30 and aggregate "how many/how much"
multi-session questions. `scripts/hindsight-record-sync` refreshes the
record. Cross-judge caveat: their verdicts came from a different judge than
ours, so agreement is evidence, not ground truth.

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
- [x] Feature `sqlite-vec` in `rustsight-storage` (default on); vec0 tables
      with write-through inserts; `search_semantic` uses KNN when enabled.
- [x] Migration/reindex command (`rustsight reindex [--bank]`).
- [x] Criterion bench: recall semantic arm at 10k / 100k vectors, brute force
      vs index, recall@10 quality delta reported
      (`cargo bench -p rustsight-storage --bench semantic`).
- [x] Graceful fallback when extension unavailable (log + brute force).
- [x] README scale guidance updated (`docs/reference.md` "Vector index and
      scale").

Done Nov 2026. Facts for later tickets:

- sqlite-vec **0.1.9** (pinned `=0.1.9`) is an *exact* SIMD-less flat
  scan in C, not an ANN: recall@10 = 1.000, single-query speedup ~2x
  (100k x 384-d: 132 → 76 ms; 50k x 1536-d: 247 → 116 ms). The cosine
  metric has no SIMD path even with `SQLITE_VEC_ENABLE_AVX` (only L2
  does), and the crate's build.rs defines neither. Retain's 20-fact
  neighbour batch is break-even (1.85 s scan vs 2.03 s KNN at 50k x
  1536-d); the index's win there is memory, not time. A real ANN (HNSW
  via usearch, or a later sqlite-vec with ANN) is the next lever and slots
  in behind `MemoryStore::search_semantic`.
- Layout (`crates/rustsight-storage/src/vector.rs`): one
  `vec_f32_<dim>` vec0 table per width, `bank_id` partition key, `kind`
  metadata column (KNN filters with `kind IN (...)`), `+ref_id` auxiliary;
  `vec_rows` maps item id → vec0 rowid for point updates/deletes.
  Extension registered per connection by calling `sqlite3_vec_init` on
  the rusqlite handle (no process-global auto-extension).
- Trust model: a bank answers from the index only while
  `vec_indexed_banks` lists it. New banks under a live index are marked
  at `ensure_bank`; any store opened without the index (`StoreOptions {
  vector_index: false }`, `RUSTSIGHT_VECTOR_INDEX=off`, feature off, load
  failure) clears all markers, so a mirror its writes skipped is never
  trusted. `reindex_vectors` clears by partition key (catches orphans
  `vec_rows` doesn't know about), re-inserts, re-marks.
- Tag scopes keep filtering before the top-k cut: KNN fetch widens x4 from
  max(4k, 64) up to vec0's k limit 4096, then falls back to the scan.
- Retain's semantic edges now come from `search_semantic` (top
  `semantic_edge_cap` per new fact, new ids excluded) instead of
  `all_embeddings`; the brute-force default still scans once per batch.

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
- [x] Compat router implementing retain/recall/reflect + bank create/update
      + documents + async-retain operations with upstream's paths and field
      names (`crates/rustsight-server/src/compat.rs`). Served at upstream's
      own paths (`/v1/default/banks/...`, `/version`, `/health`) rather than
      under `/v1/compat`, and always on rather than feature-gated: a client
      has to work with only its `baseUrl` changed, and the paths do not
      collide with the native API.
- [x] OpenAPI document generated for our native API (utoipa or hand-written
      spec) and served at `/openapi.json`. Hand-written
      (`crates/rustsight-server/openapi.json`, OpenAPI 3.1) and merged into
      the compat document already served at that path. Tests pin it to the
      router (paths and methods come from the same `native_routes()` list
      the router is built from; undocumented methods must 405), to the
      request-body structs (native bodies are `deny_unknown_fields` → 422,
      schemas `additionalProperties: false`, field lists compared via
      serde's expected-fields error), and to the core types' serialized
      fields and enum values.
- [x] Smoke test with a real Hindsight client: `@vectorize-io/hindsight-client`
      1.2.1 (TypeScript) pointed at `rustsight --mock serve` performs
      createBank → retain (plain / document+tags+metadata / append / async /
      unset timestamp) → recall (types, budget, tags, tag groups, source
      facts) → reflect (budget, context, fact types, response schema) →
      getDocument. Run by hand (Nov 2026); not yet scripted into the repo —
      the in-repo coverage is the router tests in `compat.rs`.
- [x] Compat matrix doc: supported endpoints, accepted-but-ignored options,
      behavioral differences (README "Hindsight-compatible API").

Shipped Nov 2026, driven by the first real consumer (a TypeScript agent
harness that spawns `hindsight-api` and talks to it through the official
client). Facts for later tickets:

- The engine grew what the wire shapes needed, not just a translation
  layer: documents (`document_id`, `update_mode` replace/append, stored
  `original_text`), tags with upstream's five `tags_match` modes plus
  `tag_groups`, per-memory metadata, caller-supplied entities, per-request
  reflect options (budgeted rounds, answer length, tag scope, `fact_types`,
  `response_schema`), per-operation LLMs, async retain with a persisted
  operation queue.
- Shapes were read from the installed hindsight-api **1.2.1** `api/http.py`
  (the client version the consumer runs), while prompts stay pinned at
  **1.1.0**. `/version` reports `api_version: "1.1.0"`.
- Upstream's API reflect default is `budget: low` = 5 iterations
  (`reflect_max_iterations` 10 x 0.5). The native `Engine::reflect` still
  runs `reflect_max_rounds` = 8. **The parity suite in rustsight-compare
  drives the native endpoints**, so its numbers describe an 8-round loop
  — not what a Hindsight client gets through the compat API (5 rounds at
  the default budget). Point the
  suite at `/v1/default/...` before trusting its numbers for the swap.
- Open follow-ups, in rough priority order for a real swap:
  1. Tag-partitioned consolidation (`observation_scopes`): today observation
     tags are derived from evidence and consolidation runs per entity across
     tags.
  2. Local embeddings (upstream `embeddings.provider = local`,
     bge-small-en-v1.5): rustsight needs an embeddings endpoint.
  3. `usage` token counts, reflect `trace` (tool/LLM call log). Chunk
     storage is done; chunks are still split at 3000 tokens where upstream
     uses 3000 characters.
  4. Directives as first-class resources, mental models through this API,
     HTTP MCP at `/mcp/{bank_id}/`.
  5. Delta retain: `replace` re-extracts the whole document (upstream
     reuses unchanged chunks' facts); only byte-identical re-retains are
     skipped.

---

## RS-10 — Upstream feature catch-up (triaged into RS-10a…f) · P2 · varies

### Description
Features present upstream that rustsight lacked. Triaged Nov 2026 into the
six tickets below, each with its own tests and DoD, landing as one commit
each. Upstream sources (MIT) were read from the installed hindsight-api
**1.2.1** (`api/http.py`, `extensions/memory_defense.py`, `webhooks/`,
`engine/bank_templates.py`, `config_resolver.py`, `engine/parsers/`) —
memory defense, bank templates and the webhook manager do not exist at the
1.1.0 prompt pin.

### Definition of done
- [x] Each sub-item broken into its own ticket with tests before work starts;
      this umbrella closes when triaged.

### RS-10a — Memory defense (secret/PII screening on retain) · S/M

Port upstream's regex Memory Defense: the `_REDACTION_PATTERNS` catalog
verbatim (provider keys, cloud/CI/payment/chat tokens, DB URLs, PEM, JWT,
Luhn-checked cards, US SSN) with ASCII token boundaries (so secrets inside
CJK prose still match), credit-card Luhn + UUID-overlap rejection, and
fingerprinted per-hit previews (raw values never leave the screener).
`fancy-regex` because the catalog needs look-around.

- Policy is the bank config key `memory_defense`
  (`{"enabled", "rules": [{"on": "sensitive_data", "action": "allow|redact|block"}]}`,
  upstream `parse_policy`; unknown `on` names are accepted no-ops, a bad
  action is a 4xx).
- Screening runs inside `Engine::retain` before chunking, so native, compat
  and MCP retains are all covered: `redact` rewrites the content (document
  `original_text`, chunks and facts never see the secret); `block` refuses
  the item with `Error::Blocked` (422) and stores nothing.
- Compat batches follow upstream: blocked items are dropped, an all-blocked
  batch is a 422.

DoD:
- [x] Pattern-table unit tests (one case per label, CJK boundary, Luhn
      rejects timestamps/port lists, UUID guard, fingerprint lengths).
- [x] engine_flow: redact leaves no secret in memories, chunks or document
      text; block stores nothing; disabled policy is a no-op.
- [x] compat: config accepts/validates `memory_defense`; partial-block batch
      retains the rest; all-blocked → 422.
- [x] docs/reference.md + NOTICE (1.2.1 source).

### RS-10b — Engine lifecycle events + webhooks · M

Async operations already exist (RS-9: persisted `operations`, polling,
retry). What is missing is the event side.

- `EventSink` seam on `Engine` (`with_event_sink`): `retain.completed`
  (per document), `consolidation.completed` (per bank pass with
  created/updated/deleted counts or the error), `memory_defense.triggered`
  (action, matched types, fingerprinted hits). Called synchronously at the
  emit point; embedders (Tauri) can plug their own sink.
- Webhooks (upstream `webhooks/`): `webhooks` + `webhook_deliveries`
  tables; the server's sink writes one delivery row per matching enabled
  webhook (bank-scoped rows + the global `RUSTSIGHT_WEBHOOK_URL/_SECRET/
  _EVENT_TYPES` one) — a transactional-outbox shape, so a restart resumes
  pending deliveries. Deliveries do NOT go through the operations table
  (a client waiting for "no pending operations" must not wait on a
  5-hour webhook retry).
- Delivery: upstream headers (`X-Hindsight-Event`, `-Event-Id`,
  `-Delivery-Id`, `X-Hindsight-Signature` = `X-Hub-Signature-256` =
  `sha256=<hmac body>`, `X-Hindsight-Signature-V2` = `t=..,v1=..`),
  custom headers spread first, GET/POST, params, per-webhook timeout, retry
  schedule 5s/5m/30m/2h/5h (6 attempts).
- SSRF guard: deny loopback/private/link-local/CGNAT/reserved unless
  `RUSTSIGHT_WEBHOOK_ALLOWED_HOSTS` permits the host or IP/CIDR; checked
  at registration and at delivery (resolved addresses pinned for connect).
  Response bodies withheld from the deliveries API unless
  `RUSTSIGHT_WEBHOOK_EXPOSE_RESPONSE_BODY`.
- Compat routes: `POST/GET /v1/default/banks/{id}/webhooks`,
  `PATCH/DELETE …/webhooks/{webhook_id}`, `GET …/{webhook_id}/deliveries`.

DoD:
- [x] engine_flow: a capturing sink sees retain, consolidation and defense
      events with the upstream payload fields.
- [x] server tests: CRUD + validation (SSRF, unknown event type), signed
      delivery to a local receiver (allowlisted), retry scheduling after a
      500, deliveries listing.
- [x] docs/reference.md env vars + compat matrix.

### RS-10c — File ingest + attachments · M

Upstream `POST /v1/default/banks/{id}/files/retain` (multipart `files` +
`request` JSON with `parser` / `files_metadata`), always async — one
operation per file (`file_convert_retain`).

- `attachments` table (id, bank, document, filename, content type, size,
  sha256, bytes); `GET /v1/default/banks/{id}/attachments/{attachment_id}`
  serves the original bytes. Deleting the document or bank deletes them.
- `FileParser` trait + registry in core; built-ins: `text` (UTF-8 text,
  markdown, CSV/JSON/YAML/XML, source code) and `html` (tag-stripped
  text). Parser chains fall back in order; requesting an unregistered
  parser is a 400 (upstream allowlist semantics). Heavier converters (PDF,
  DOCX, OCR) slot in behind the trait later.
- The extracted text retains under the file's document with the file's
  tags/context/timestamp; every fact carries `attachment_id` (and
  `filename`) in its metadata.
- Limits (upstream defaults): 10 files, 100 MB per request
  (`RUSTSIGHT_FILE_CONVERSION_MAX_BATCH_SIZE[_MB]`), endpoint toggle
  `RUSTSIGHT_ENABLE_FILE_UPLOAD_API` (default on), default chain
  `RUSTSIGHT_FILE_PARSER` (default `text,html`).
- CLI: `rustsight retain --file <path>` runs the same parse locally.

DoD:
- [x] Parser unit tests (text decode, HTML strip incl. script/style, chain
      fallback, unsupported type error).
- [x] compat: multipart upload → operation completes → memories carry
      `attachment_id`; attachment download returns the bytes; bad parser /
      too many files / metadata count mismatch → 400.
- [x] Storage: attachment rows die with their document and bank.

### RS-10d — Bank templates · S/M

Upstream `engine/bank_templates.py`: manifest `{version: "1", bank,
mental_models, directives}`.

- `POST /v1/default/banks/{id}/import[?dry_run=true]`,
  `GET /v1/default/banks/{id}/export`, `GET /v1/bank-template-schema`;
  CLI `rustsight template import|export`.
- `bank` applies the config keys this server acts on (same set as
  `PATCH /config`); any other non-null upstream field is refused by name
  rather than silently dropped.
- Directives become first-class (name, content, priority, is_active,
  tags) — stored per bank, matched by name on re-import, active ones
  rendered into the reflect prompts by priority. The legacy single
  `Bank.directives` string keeps working.
- Mental models are matched by id (`source_query` → question, `name`
  stored); content fills on the next refresh.
- `RUSTSIGHT_DEFAULT_BANK_TEMPLATE` (inline JSON or a file path) is applied
  to every newly created bank.

DoD:
- [x] Unit: manifest validation (version, duplicate ids/names, unsupported
      fields).
- [x] compat: import → export round-trip; dry run changes nothing;
      re-import updates by id/name.
- [x] engine_flow: imported directives reach the reflect prompt; default
      template lands on a bank created by retain.

### RS-10e — Multilingual preservation · S/M

Extraction already carries upstream's preserve-the-source-language rule
(`default_language_section`) and the output-language directive builder;
neither the setting nor CJK search was wired.

- `RUSTSIGHT_LLM_OUTPUT_LANGUAGE` (+ `HINDSIGHT_API_LLM_OUTPUT_LANGUAGE`)
  feeds extraction, consolidation and reflect prompts; unset keeps the
  source-language rule (facts untranslated, entities in native script).
- FTS: CJK runs (Han, Kana, Hangul) are indexed and queried as
  overlapping bigrams next to the unicode61 tokens (Lucene CJKAnalyzer
  shape) — `porter unicode61` alone makes a whole CJK sentence one token.
  A one-time migration re-indexes existing rows.
- Script detection (`lang.rs`) routes queries: Han-bearing queries also
  run a Chinese relative-period parser (今天/昨天/前天, 本周/上周/下周,
  本月/上个月, 今年/去年/前年, 最近/过去 N 天/周/个月/年, N 天前,
  YYYY年M月[D日], M月D日) — the common subset of upstream's
  `chinese_temporal_periods.py`; the full rule set is a recorded
  divergence.

DoD:
- [x] Unit: bigram tokenization, script detection, Chinese period table.
- [x] engine_flow: Chinese content retained with native-script entities;
      a Chinese keyword query finds it through BM25; `上周` narrows the
      temporal arm; output language reaches the extraction prompt.
- [x] Storage: migration test opening a pre-bigram FTS file.

### RS-10f — Hierarchical config (global → tenant → bank) · M

Upstream `config_resolver.py`: global env → tenant (extension) → bank
overrides, resolved per request.

- A tenant here is a named config layer (`tenants` table, id + JSON); a
  bank joins one via its `tenant` setting (or `RUSTSIGHT_DEFAULT_TENANT`).
  No auth-scoped isolation — that stays the single-token model.
- One registry of configurable keys: engine knobs (chunk size/overlap,
  reflect rounds/answer length, recall budget levels, rerank cap,
  consolidation window/dupe threshold, output language) and bank settings
  (missions, observations toggle, labels, memory defense). Values are
  type-checked on write at either layer.
- `Engine::resolved_config(bank)` merges the layers; retain, recall,
  reflect and consolidation read their knobs and settings through it
  instead of the global `EngineConfig`.
- Native API: `GET/PATCH /tenants/{id}/config`, `GET /banks/{id}/config`
  (resolved values + which layer each came from); CLI `rustsight tenant`.

DoD:
- [x] Unit: layer precedence, type validation, unknown keys refused.
- [x] engine_flow: a tenant chunk-size override changes retain chunking for
      its banks only; a bank override beats the tenant.
- [x] HTTP: tenant config round-trip; resolved view reports provenance.

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

## RS-14 — Upstream-parity entity resolution + recall pool (done) · P0 · M

### Description

The fullstack smoke run failed its only knowledge-update question
(`fss_q4_update`: answered Globex, gold Anthropic). Postmortem: the person
fragmented into three entities (`Alice Chen` / `Alice` / `Dr. Chen`) because
rustsight's old resolver math could never link a given-name mention to a
full-name entity (`0.6·str + 0.2·co + 0.2·temp` vs threshold 0.82 caps
"Alice"→"Alice Chen" at 0.76), and the job-change fact then never reached
reflect (top-10-per-arm pool, graph arm unreachable across entities). Fix =
exact port of upstream v1.1.0's resolver and pool sizing, no invented
constants.

### Need-to-know facts

- Upstream resolver (`memories/pg/entity_resolver.py`, verified at the pin):
  trigram floor 0.3 → word-compat gate (prefix/seq≥0.6 per token, numbers
  rule) → `0.5·SequenceMatcher + 0.3·co-occurrence(1/√degree-damped,
  per-fact nearby) + 0.2·recency(7-day window)` vs inclusive 0.6 (round 6);
  identical trigram sets score 1.0; in-batch union-find clustering of new
  names at 0.5 + word-compat, canonical = most-mentioned→shortest→lex,
  cap 250; co-occurrence matched by name against per-fact sibling sets.
- Replaying that exact math on the smoke fixture still fragments (≈0.56 <
  0.6 — one damped shared partner isn't enough corroboration); upstream
  answers this question class through recall robustness: reflect's recall
  pool is budget-driven (None → MID → fixed 300 per method per fact_type,
  `GRAPH_SEED_LIMIT` 20) and reranked with combined scoring (sigmoid CE ×
  ±10% recency × ±5% proof-count, linear 365-day decay, 0.1 floor). Both
  ported here.
- `string_similarity` (Levenshtein) is NOT part of the resolver anymore but
  stays: chunk overlap dedup (0.9) and consolidation observation dedup
  (0.85) use it.

### Definition of done

- [x] `trigram.rs` port (pg_trgm word trigrams, Jaccard) with upstream's
      calibration cases pinned in tests.
- [x] `entity.rs`: difflib `SequenceMatcher.ratio` port (unit-tested against
      Python-generated values), token gates, `score_candidate`,
      `cluster_new_entity_names`.
- [x] `retain.rs`: per-fact nearby scope (kills alias-cache order
      dependence), whole-bank `CooccurrenceIndex` snapshot pre-link,
      in-batch clustering with find-or-create canonicals,
      `link_memory_entities(…, at_ms)` advancing last-seen by the fact's
      event date.
- [x] Recall: `RecallBudgetConfig` (fixed/adaptive, upstream defaults),
      graph seeds 20, uncapped rerank (`reranker_max_candidates` 0),
      `apply_combined_scoring` port incl. passthrough RRF seeding
      (`RerankProvider::is_passthrough`; `NoopRerank` returns no scores).
- [x] Tests: upstream resolver cases (John/Jane Smith, Tigran/Iran, UA123/
      124, Wren emoji, Merrivale typo, degree damping, 7-day cutoff) +
      engine_flow merge-positive and fragment+wide-pool integration tests.
- [x] Offline `--mock` fullstack smoke 7/7; real run pending (user).

## RS-17 — Parity-run iteration: reflect answer machine + consolidation response machine (done) · P1 · L

### Description

The first hosted parity-suite run vs Hindsight (2026-11-11, gpt-oss-120b via
OpenRouter, 6 scenarios / 20 questions) scored 40% overall: 6 `rustsight_less`
(broad list questions answered from a fraction of what was retrieved — e.g.
"What does Joe hate?" retrieved 150 items, cited 3), 2 contradictions
(consolidation baked first-person inferences into observations — "Joe …
participated"; reflect inferred "likes hibachi/thai/pierogi" from outing
memories), and 2 whole scenarios dead with `reflect produced no answer` HTTP
500s (the loop searched every round and never called `finish`). Root causes
were structural, not prompt-wording: observations were 6–8x upstream's count
(196/245 vs 31/26) with verbatim cross-entity duplicates crowding reflect's
token budget, the recall pool was observation-dominated (upstream's reflect
recall returns world/experience only), and the loop model was asked to both
stop and compose the final answer — upstream throws that answer away and
re-synthesizes from the full gathered history.

### Shipped (Nov 2026)

- Reflect answer machine: kind-separated tools (`recall` → world/experience,
  `search_observations` → observations, opinions unretrievable by reflect),
  forced-done round on round-exhaustion/empties, then a separate
  final-synthesis call over the packed tool history (`build_final_prompt` /
  `split_context_history` / claims-map + reduce, hand-ported from the vendored
  upstream). "reflect produced no answer" is now reachable only when the
  synthesis provider call fails with no fallback.
- Consolidation response machine: byte-parity upstream prompt with
  per-entity `ENTITY:`-scoped batches, creates/updates/deletes application
  (deletes merge evidence into `merged_into` survivors), bank-level
  exact-dupe guard, update re-embedding, and a lenient parser accepting the
  legacy `{"observations": [...]}` shape.
- Store: `delete_observation` (row + FTS).

### Definition of done

- [x] No scenario errors from model-shape failures (forced-done + synthesis).
- [x] `ReflectResponse.synthesized` records the answering path; eval traces it.
- [x] engine_flow: upstream-plan application, delete-merge, dupe guard,
      kind separation (synthesis prompt capture), round-exhaustion,
      split-synthesis, empty-streak synthesis tests.
- [x] `cargo fmt` / `clippy -D warnings` / all tests green; offline
      fullstack smoke 7/7.
- [x] Parity-suite rerun (Nov 2026, reranker on CUDA): **regressed to 13.8%**
      (4/29 equivalent, 15 `rustsight_less`, 4 contradictions, 0 errors —
      run `20261111T195956Z` in rustsight-compare). The synthesis-first
      answer machine writes minimal literal answers even with complete
      evidence in the packed history (alice_profile: cited observation said
      "four years on the Bing ranking pipeline"; answer said "four years").
      Root causes and follow-ups ticketed as RS-18..RS-21 below.
- [ ] Golden-pin the hand-ported final-synthesis builders via
      `scripts/prompt-goldens` (extend the generator's harness) — rolled
      into RS-18's DoD.

Facts for the rerun: answers now cost one extra LLM call per reflect (the
synthesis; upstream pays it too) — expect reflect wall time up ~50%, answer
completeness up. The escape-room contradiction may persist even after the
consolidation fix: our observation asserted what the first-person journal
implied, gold says otherwise — partly judge/gold noise.

## RS-15 — Full TEMR recall parity · P1 · L

Upstream recall is a budget-driven pipeline across per-fact_type arms with
strategy boosts, min-score floors, `interleave`/`rrf`/`cross_encoder`
reranking modes, recency decay config plumbing, and per-level
`reranker_max_candidates`. rustsight's recall is one pool per arm across the
requested kinds. Port the structure (per-kind per-arm limits, budget levels plumbed
through the reflect tool schema, decision-reranker hook) — see
`memories/pg/recall.py`, `search/retrieval.py`, `search/fusion.py`,
`memory_engine.py` at the pin. The reflect-side kind alignment (recall →
world/experience, observations behind `search_observations`) shipped with
RS-17; what remains is the *engine* recall's per-fact_type arm structure.

## RS-16 — Entity merge/alias repair API (upstream #3216) · P2 · M

Neither rustsight nor upstream (issue #3216, open) can repair an already
fragmented bank: a desktop bank retained before RS-14 keeps its duplicate
entities forever. Add a merge verb (canonical entity absorbs another's
aliases, memory links, observations, co-occurrences; dirty queue re-runs
consolidation) plus `inspect` reporting of likely duplicate pairs
(`cluster_new_entity_names` on existing entities as the candidate
generator).

## RS-18 — Reflect answer path back to upstream semantics + evidence breadth · P0 · L

### Description

The Nov 2026 parity rerun (`20261111T195956Z`, gpt-oss-120b via OpenRouter,
6 scenarios / 29 questions, reranker on CUDA) regressed to 13.8% parity from
the first run's 40%, all failure modes being answer *depth*, not retrieval:

- 15/29 `rustsight_less`. Retrieval is NOT the bottleneck: alice_profile
  "How long did Alice work at Microsoft?" retrieved 6/6 bank observations,
  the top hit was "Alice Chen worked at Microsoft for four years on the
  Bing ranking pipeline", and the answer still came back "Alice worked at
  Microsoft for **four years**." role_change "Which company does Bob work
  for?" retrieved the whole Stripe→Figma timeline; answer: "Bob works for
  **Figma**." The separate final-synthesis call (RS-17) writes minimal
  literal answers even with complete evidence in its packed history.
- Upstream's reflect does the opposite of what we shipped: the loop's
  `done` tool call IS the answer (verified in installed hindsight-api
  1.1.0 `agent.py`: the done payload is validated, id-stripped, and
  returned; `build_final_prompt` is only the *fallback* when the closing
  done call fails — `_ask_for_done`'s except path). Gold answers show the
  loop searched both levels (they cite observation AND raw-fact ids with
  observed/recalled dates), i.e. the ladder + done answer written with the
  whole tool conversation in context.
- Our loop prompt also lacks the two upstream behaviors that drive gold's
  evidence breadth: the **Query Strategy** section ("NEVER just echo the
  user's question — decompose it into targeted searches … search each
  entity/concept separately", vendored `reflect/prompts.py`
  `build_system_prompt_for_tools`) and the **done sufficiency gate**
  (`_DONE_SUFFICIENCY_GATE` + `answer_found` bool in `tools_schema.py`,
  with the #4567 guardrail: refuse a negative `done` until recall ran).
  With gpt-oss batching `search_observations`+`finish` in one round, most
  reflects gather a single literal 2000-token search then answer.
- Tool budget defaults diverge: upstream search_observations defaults to
  5000 result tokens (recall: 2048); ours say 2000
  (`prompts/reflect.rs::reflect_tools`, `engine/reflect.rs` default).

### Work

1. [x] **Done-first answer priority** in `engine/reflect.rs`: a non-empty loop
   `finish` answer (including the finish-shaped-content unwrap and the
   forced-done round) becomes the response; the standalone synthesis runs
   only when the loop produced nothing usable (round exhaustion, empties,
   provider failure) — upstream's exact precedence.
   `ReflectResponse.answer_path` records the path (loop | forced-done |
   synthesis); `synthesized` stays as the compatibility view. One fewer
   LLM call on the common path.
   **Iteration 2 (after the 2026-11-12 rerun, 13.8% → 20.7% but still
   12/29 `rustsight_less` + 1 abstain):** the closing ask
   (`build_done_request_prompt` + forced `finish` via the new
   `LlmRequest.tool_choice`, upstream `tool_choice=done`) now ALWAYS runs
   when the loop stops and its answer wins (ForcedDone); the volunteered
   finish is the fallback (Loop); synthesis stays the last resort. Also:
   harness-noise content (`<|channel|>`-style tokens) is treated as an
   empty turn (one answer was raw gateway garbage), and every answer is
   stripped of inline memory ids / "Citations" sections with the ids
   harvested into the validated citations (upstream
   `presenter.drop_unseen_ids`).
   **Iteration 3 (after the 2026-11-12 31.0% run):** the missing depth
   lever was mechanical, not luck — upstream `reflect_async` defaults
   `max_tokens: int = 4096` (a *visible answer-length target*, distinct
   from the output cap), which rides the closing ask and synthesis
   prompts as the `## Length` directive ("a complete, self-contained
   answer of approximately N tokens"). We passed `None`, so the same
   model that writes gold's exhaustive answers wrote 41-char one-liners
   on point questions with perfect retrieval (alice flipped 100% → 25%
   across runs on this alone). New
   `reflect_answer_max_tokens` config (default 4096) now feeds
   `build_done_request_prompt`, `build_final_prompt`, and
   `build_reduce_prompt`.
2. [x] **Port the upstream loop guidance into our diverged system prompt**
   (`build_reflect_system`): the Query Strategy section (verbatim where
   possible, tool names adapted), the research-depth (MID) wording.
3. [x] **Sufficiency gate on `finish`**: `answer_found` (bool) on the tool
   schema with upstream's description + `_DONE_SUFFICIENCY_GATE`
   (done→finish) appended to the description; `answer_found=false` before
   any `recall` is refused once with a nudge, the retry honored (applies
   to finish tool calls and finish-shaped content; not the forced-done
   closing ask, matching upstream).
4. [x] **Tool budget defaults** → search_observations 5000, recall 2048
   (schema defaults and `run_recall_tool` fallbacks).
5. [ ] Update DIVERGENCES.md item 5 (done — see above) and golden-pin the
   hand-ported final-synthesis builders (RS-17 leftover) via
   `scripts/prompt-goldens` — the generator extension is still open.

### Definition of done

- [x] `answer_path` recorded and asserted in engine_flow tests: loop
      finish preferred (`reflect_makes_no_synthesis_call_when_loop_finishes`,
      `reflect_loop_with_scripted_tools`); synthesis only on empty loop;
      forced-done answer wins when it lands (eval runner test).
- [x] Sufficiency-gate refusal tests:
      `reflect_refuses_premature_negative_finish_until_recall`,
      `reflect_honors_repeated_negative_finish_without_recall`.
- [x] Prompt tests pin the new Query Strategy / depth sections and tool
      budget defaults (`reflect_system_carries_query_strategy_and_depth`,
      `reflect_tool_budget_defaults_match_upstream`,
      `finish_tool_carries_sufficiency_gate_and_answer_found`).
- [x] fmt / clippy -D warnings / 228 tests green / offline fullstack
      smoke 7/7.
- [x] First parity rerun (2026-11-12): 20.7% (from 13.8%), less 12 (from
      15), contradictions 2 (from 4), first exact-string match — but under
      the ≥40% target; iteration 2 (closing ask + hygiene) shipped on the
      same ticket.
- [ ] Second parity rerun: `rustsight_less` ≤7/29, overall parity ≥40%;
      0 errors.

## RS-19 — Consolidation completeness: fact cap + near-dupe observations · P0 · M

### Description

Two consolidation defects keep observations weaker than gold's:

- **60-fact cap drops facts silently.** `consolidate.rs` feeds each dirty
  entity `facts_for_entity(eid, 60)` (newest-first). The 2025-11 scenario
  has 159 facts — the "Joe" entity alone exceeds 60 — so older facts never
  reach an observation on any pass (new facts keep pushing them out of the
  window). Reflect must then find them via raw recall, which list-style
  questions rarely do.
- **Near-duplicate observations still crowd the bank.** 2025-11: 199
  observations vs Hindsight's 31 (role_change: 5 vs 3). The exact-text
  dupe guard can't catch paraphrases, and per-entity batches never show
  the model observations outside the entity's own set, so it CREATEs a
  sibling instead of UPDATEing (role_change: "Bob began his career at
  Stripe as a junior developer on the payments team on March 2, 2026" and
  "Bob started as a junior developer on the payments team at Stripe."
  both live on, linked to different entities). Upstream pools existing
  observations bank-wide via recall per batch (`consolidator.py`, not
  vendored — see DIVERGENCES item 4).

### Work

1. [x] Page through ALL of a dirty entity's facts in consolidation windows
   (`consolidate_max_facts_per_entity`, default 200, newest-first, later
   windows see earlier windows' observations; one batched embed per
   entity). The old fixed cap of 60 silently dropped older facts of big
   entities.
2. [x] Cosine near-dupe guard at CREATE time (`consolidate_dupe_cosine`,
   default 0.9): a proposed observation this similar to any embedded
   existing observation merges its evidence instead of writing a sibling;
   exact-text guard stays the zero-cost first pass; create texts are
   embedded once (batched) and reused for the stored embedding.
3. [ ] (Measured-optional) widen the consolidation prompt's existing
   observations with top-k recalled bank-wide observations, upstream-style
   — only if the rerun still shows cross-entity paraphrase siblings.

### Definition of done

- [x] engine_flow: entity with more facts than one window consolidates
      them all (`consolidation_pages_past_the_fact_cap`); nothing silently
      dropped.
- [x] engine_flow: a CREATE paraphrasing an existing observation (cosine ≥
      threshold) merges evidence instead of writing a sibling; a genuinely
      new CREATE is untouched
      (`consolidation_cosine_guard_merges_paraphrase_siblings`).
- [x] fmt / clippy / 233 tests / fullstack smoke 7/7 (rotation bug in the
      smoke script fixed too).
- [ ] Parity-suite rerun: observation count for the journal scenarios
      within ~2x Hindsight's (from ~5-8x); no parity regression.

## RS-20 — Inference discipline for contradictions · P1 · S

### Description

All 4 contradictions are synthesis-level over-inference, not bad data:

- "What did Amy do in January?" added trauma-dumping, which the bank only
  records as *mentioned* in January ("Sam's behavior reminds him of Amy's
  trauma-dumping when Joe asked for a divorce") — occurred years earlier.
- "What food does Joe likes?" listed outing foods as preferences; gold
  abstains ("no statement about specific foods Joe likes").
- "What days did Joe see Emma?" counted co-writing/planning days as
  in-person meetings; gold lists only the documented visit.
- "Who went to the escape room?" added Joe (first-person journal "escape
  room with Meredith, Andrew, Jay, and Emma") — partly gold noise (the
  journal voice does imply Joe attended; RS-17 noted this), acceptable to
  lose.

RS-18's done-first change may fix some of this for free (gold's strictness
came from a loop that had run raw `recall`, whose literal facts — "wants
to get pierogi during Pierogi week" — invite less inference than our
distilled observations). Do this ticket only for what survives that rerun.

### Work

1. [x] (shipped with RS-19's round) Mentioned-vs-occurred + participation/
   preference inference rules added to the loop system prompt as an
   engine-side **Evidence Discipline** section (extends upstream's
   grounding boundary; the closing ask rides the same conversation, so
   the rules are in scope when the answer is written). Wording encodes
   the bank owner's adjudicated semantics (2026-11-12): together
   activities (co-writing) ARE meetings; planning/scheduling is not;
   eating/going somewhere is not liking it; one explicitly liked meal is
   not a standing preference; "in <period>" selects on occurred dates;
   before/after-a-date questions weigh every milestone (the Atlas beta
   case). Note: the Emma-days parity verdicts will read as
   `contradiction` while rustsight is the *more accurate* side there —
   the adjudication is recorded in rustsight-compare's README.
2. [ ] If preference/attendance inference survives the RS-19 rerun,
   extend the synthesis system prompt's grounding boundary the same way
   (it only matters on the synthesis fallback path now).
3. [x] Documented in DIVERGENCES.md item 5 as engine-side additions.

### Definition of done

- [x] Prompt tests pin the new wording
      (`reflect_system_carries_evidence_discipline`).
- [ ] Parity rerun: contradictions ≤3/29 (escape-room expected to persist
      as gold noise; Atlas is borderline gold noise), no new abstain
      mismatches.

## RS-21 — Latency: opinion extraction off the reflect critical path · P1 · S

### Description

Every reflect awaits `form_opinions` (one extra LLM call per question)
before responding: 29/29 reflects in the parity run formed **zero**
opinions, while paying roughly 5–15 s each (reflect avg 35 s vs
Hindsight's 20 s; loop + synthesis + opinion extraction = 3+ calls per
question). The journal scenarios' answers simply don't carry first-person
opinions.

### Work

1. [x] Move `form_opinions` off the response path: spawned as a background
   task after the answer is composed (errors logged, never blocking the
   HTTP response; `with_retry_config` carries the engine's retry policy
   into the detached task). Tests poll the store for the landed opinion
   (worker-test pattern).
2. [x] Per-call latency splits in the reflect response:
   `loop_ms` (all rounds + forced-done) and `synthesis_ms` (when it ran);
   `rounds` already existed.
3. [x] Re-ask visibility: `RetainResponse.reasks` counts half-size
   re-asks per retain; a one-line warn logs each fire. The parity driver
   records the response JSON, so per-scenario re-ask totals fall out of
   the run. Open follow-up if re-asks dominate retain latency:
   structured-output (`json_schema`) support in `OpenAiCompat` for
   extraction (upstream uses response-model schemas) — should cut both
   latency and parse failures.

### Definition of done

- [x] HTTP reflect returns without waiting on opinion extraction;
      engine_flow proves opinions still form in the background
      (`reflect_forms_and_stores_opinions`,
      `opinion_linked_to_entity_is_reinforced_by_consolidation`).
- [x] Reflect metadata carries per-phase timings
      (`loop_ms`/`synthesis_ms`); assert via any reflect response.
- [x] Re-ask counter exists and is asserted
      (`retain_reasks_at_half_size_on_parse_failure`).

## Appendix — facts referenced by multiple tickets

- Upstream pin: `vectorize-io/hindsight` tag `v1.1.0`, commit `a5a619b5`,
  MIT (root LICENSE, Copyright 2025 Vectorize AI, Inc.). Engine source path:
  `hindsight-api-slim/hindsight_api/engine/`.
- Paper: arXiv:2512.12818, TEMPR (retain/recall) + CARA (reflect); four
  networks W/B/O/S; RRF + cross-encoder + token budget; LongMemEval-S
  reference scores 83.6 / 89.0 / 91.4 (OSS-20B / OSS-120B / Gemini-3 head).
- Current test baseline: 39 tests green, clippy clean (as of initial commit
  `9070afb`).
- Deviations already accepted: exact (non-ANN) vector KNN (RS-7), informed-rewrite
  prompts (RS-5), sync-only consolidation (RS-2), custom HTTP shapes (RS-9).
