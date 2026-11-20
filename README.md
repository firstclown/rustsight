rustsight

A Hindsight-style agent memory engine written in Rust, designed as an embeddable
component for desktop applications. Implements the architecture of "Hindsight is
20/20: Building Agent Memory that Retains, Recalls, and Reflects"
(arXiv:2512.12818) with prompts informed by the upstream MIT-licensed
vectorize-io/hindsight implementation.

Architecture
------------
Four memory networks, three operations:

- retain: LLM narrative fact extraction (temporal ranges, typed entities,
  causal links) -> entity resolution (upstream v1.1.0's exact fuzzy model:
  pg_trgm trigram floor + word-by-word gate, 0.5·name + 0.3·co-occurrence +
  0.2·7-day-recency vs 0.6, plus in-batch clustering of new names) ->
  typed graph edges (entity / temporal / semantic / causal) -> dirty-entity
  queue
- recall: four parallel retrieval arms (vector cosine, FTS5 BM25, 2-hop graph
  traversal, parsed temporal window) -> reciprocal rank fusion -> cross-encoder
  rerank with combined scoring (recency + proof-count boosts) -> greedy
  token-budget cut; reflect's pool width is budget-driven (fixed MID = 300
  per arm, upstream defaults)
- reflect: agentic tool loop (search_observations / recall / read_observation /
  finish) shaped by the bank's mission, directives, and disposition traits
  (skepticism, literalism, empathy, bias strength). The loop's `recall` returns
  raw world/experience facts only; observations are searched separately
  (upstream's kind-separated ladder), with query-decomposition guidance, the
  #4567 sufficiency gate on premature "nothing known" finishes, and upstream
  tool budget defaults. Upstream's answer precedence: when the loop stops, a
  closing ask (`_ask_for_done`) forces the `finish` over the full tool
  conversation and that answer wins; the loop's volunteered finish is the
  fallback, and a separate final-synthesis call (map-to-claims/reduce for
  oversized histories) is the last resort. Answers are stripped of inline
  memory ids (harvested into citations); harness-noise content never becomes
  an answer. Opinion formation runs in the background; citations are
  validated against retrieved ids

Background consolidation turns facts into evidence-backed observations;
mental models are standing answers refreshed from the bank.

Storage: a single SQLite file (WAL, FTS5 for BM25, BLOB embeddings scanned by
exact cosine - appropriate at desktop scale). Swap-in points for ANN indexes
and Postgres exist via the `MemoryStore` trait.

Crates
------
- rustsight-core      types, provider traits, prompts, pipelines
- rustsight-storage   SQLite implementation of `MemoryStore`
- rustsight-providers OpenAI-compatible LLM/embeddings/rerank over HTTP, mocks
- rustsight-server    axum HTTP API + stdio MCP server
- rustsight-cli       CLI binary (`rustsight`)

Build
-----
    cargo build                 # debug (workspace)
    scripts/build-release       # release binary with the local ONNX reranker
                                # (CUDA; --cpu for CPU-only) — required for
                                # RUSTSIGHT_RERANK=local. A plain
                                # `cargo build --release` omits the reranker.
    cargo test

Configuration (non-mock mode)
-----------------------------
    RUSTSIGHT_LLM_BASE_URL    default https://api.openai.com/v1
    RUSTSIGHT_LLM_API_KEY     required unless --mock
    RUSTSIGHT_LLM_MODEL       default gpt-4o-mini
    RUSTSIGHT_RETAIN_LLM_*    optional per-operation model: _BASE_URL / _API_KEY /
    RUSTSIGHT_REFLECT_LLM_*   _MODEL for retain (extraction), reflect, and
    RUSTSIGHT_CONSOLIDATION_LLM_*  consolidation. Unset parts inherit the main
                              LLM's — e.g. extract with a small model and
                              answer with a large one
    RUSTSIGHT_EMBED_BASE_URL  defaults to RUSTSIGHT_LLM_BASE_URL
    RUSTSIGHT_EMBED_MODEL     default text-embedding-3-small
    RUSTSIGHT_HTTP_TIMEOUT_SECS  total wall for one provider HTTP round trip,
                              connect through last body byte (default 300,
                              clamped 5-3600). Transport-level failures
                              (dropped connection, timeout) are retried like
                              429/5xx
    RUSTSIGHT_RETAIN_MAX_COMPLETION_TOKENS  output cap per fact-extraction
                              call (default 64000, as in Hindsight; lowered
                              automatically for gpt-4o and gpt-4.1). Lower it
                              if your model rejects the default
    RUSTSIGHT_RERANK          local | api | off | auto (default auto; see below)
    RUSTSIGHT_RERANK_URL      optional; any /rerank endpoint (Jina/Cohere style)
    RUSTSIGHT_RERANK_MODEL_PATH  pre-placed ONNX model (file or dir); skips download
    RUSTSIGHT_RERANK_MODEL_SHA256  optional; hex sha256 pin — a downloaded model
                              file whose hash differs is refused
    RUSTSIGHT_RERANK_DEVICE   auto | cpu | cuda (default auto)
    RUSTSIGHT_DB              default ~/.rustsight/rustsight.db
    RUSTSIGHT_ASYNC_RETAIN_CONCURRENCY  async retains run at once through the
                              Hindsight API (default 2, range 1-64)
    RUSTSIGHT_API_TOKEN       optional; bearer token required by every HTTP
                              endpoint except /healthz (see below)
    RUSTSIGHT_API_TOKEN_FILE  optional; read the token from a file instead

Every LLM setting above also answers to its `HINDSIGHT_API_*` name
(`HINDSIGHT_API_LLM_BASE_URL` / `_API_KEY` / `_MODEL`, and the
`HINDSIGHT_API_RETAIN_LLM_*`, `_REFLECT_LLM_*`, `_CONSOLIDATION_LLM_*`
families), so a supervisor that launches `hindsight-api` with that
environment can launch rustsight unchanged; the `RUSTSIGHT_*` name wins
when both are set. Embeddings and reranking are *not* aliased — see the
compatibility notes below.

Hindsight-compatible API (drop-in for Hindsight clients)
--------------------------------------------------------
`rustsight serve` also speaks the upstream `hindsight-api` wire protocol, so
the official clients (`@vectorize-io/hindsight-client`,
`hindsight-client` for Python) work against it unchanged — point their
`baseUrl` at the rustsight server. Shapes follow hindsight-api 1.2.x; errors
use FastAPI's `{"detail": "..."}` body. The native API (`/banks/...`) is
untouched and served side by side.

    GET    /health, /version, /openapi.json                (public)
    GET    /v1/default/banks
    PUT    /v1/default/banks/{bank_id}                     create/update (PATCH too)
    DELETE /v1/default/banks/{bank_id}                     delete bank and contents
    GET    /v1/default/banks/{bank_id}/profile
    GET    /v1/default/banks/{bank_id}/config              (PATCH, DELETE too)
    POST   /v1/default/banks/{bank_id}/memories            retain (sync or async)
    POST   /v1/default/banks/{bank_id}/memories/recall
    POST   /v1/default/banks/{bank_id}/reflect
    GET    /v1/default/banks/{bank_id}/documents
    GET    /v1/default/banks/{bank_id}/documents/{id}      (DELETE too)
    GET    /v1/default/banks/{bank_id}/operations          list, newest first
    GET    /v1/default/banks/{bank_id}/operations/{id}     async retain status
    POST   /v1/default/banks/{bank_id}/operations/{id}/retry

`rustsight serve --host 127.0.0.1 --port 8888 --log-level warning` accepts
the `hindsight-api` command line, and the server drains on SIGTERM as well
as ctrl-c.

What works the Hindsight way:

- **Retain**: `items[]` with `content`, ISO `timestamp` (or `"unset"`),
  `context`, `metadata`, `tags`, `entities`, `document_id`, `update_mode`
  (`replace` deletes the old version's memories and re-extracts; `append`
  extends the stored body and extracts only the new tail). Re-retaining a
  document byte-for-byte is a no-op. `async: true` returns an
  `operation_id` immediately; queued work survives a restart, and
  `RUSTSIGHT_ASYNC_RETAIN_CONCURRENCY` sets how many run at once.
- **Recall**: `types`, `budget` (low/mid/high → 100/300/1000 per arm),
  `max_tokens` (default 4096), `tags` + `tags_match`
  (any / all / any_strict / all_strict / exact), `tag_groups`
  (and / or / not, with `resolve: "fuzzy"` leaves), `query_timestamp`,
  `prefer_observations`, `min_scores.final`, `include.entities`,
  `include.source_facts`, `include.chunks`, `trace`. Results carry `id`,
  `text`, `type`, `entities`, `context`, `metadata`, `tags`, `document_id`,
  `chunk_id`, ISO `occurred_*` / `mentioned_at`, `scores.final`.
- **Reflect**: `budget` (low/mid/high → 5/10/20 loop rounds; default low),
  `max_tokens`, `context`, `tags` / `tags_match` / `tag_groups`,
  `fact_types`, `response_schema` → `structured_output`,
  `include.facts` → `based_on.memories`.
- **Banks**: `name`, `reflect_mission` (aliases `mission`, `background`),
  `disposition`, `retain_mission`, `observations_mission`,
  `enable_observations`, `entity_labels`, `entities_allow_free_form`. A bank
  configured through this API keeps its mission for reflect only, as
  Hindsight does. `PATCH .../config` takes those settings under `updates`
  (plain or `HINDSIGHT_API_*` names) and refuses any other key. Deleting a
  bank cancels its in-flight retains and consolidation.
- **Operations**: list by `status` / `type`, retry failed retains. While
  consolidation has queued work, it appears as one pending `consolidation`
  operation, so waiting for zero open operations includes observations.
- **Documents**: fetch (`original_text`, `memory_unit_count`, tags,
  metadata), list, delete (cascades to the document's memories).

Accepted but not acted on (the request succeeds, the option is ignored):
`observation_scopes`, `strategy`, `resolve_entities`, `min_scores.semantic` /
`keyword` / `reranker`, `rerank.prune` / `dedup`, `temporal_window`,
`exclude_mental_models*`, `apply_all_directives`, `include.tool_calls`
(`trace` is null), the retain-tuning and retrieval-toggle fields of bank
create. `usage` is always null. Rejected with a 400: image/file content
blocks.

Behavioral differences to know before swapping:

- **Observation scoping is derived, not partitioned.** An observation's tags
  are the union of its evidence facts' tags and tag filters apply to that
  union; consolidation itself still runs per entity across tags
  (`observation_scopes` is ignored). Within one bank, an observation can
  therefore draw on facts from several tag scopes. Use one bank per
  tenant — as Hindsight's own guidance suggests — if tags separate users.
- **Embeddings are API-only.** Hindsight's local embedding provider
  (`HINDSIGHT_API_EMBEDDINGS_PROVIDER=local`) has no equivalent; set
  `RUSTSIGHT_EMBED_*` to an OpenAI-compatible embeddings endpoint.
- **Reranker** is configured with `RUSTSIGHT_RERANK*` (see below), not
  `HINDSIGHT_API_RERANKER_*`.
- **Storage is one SQLite file** (`RUSTSIGHT_DB`), not Postgres; existing
  Hindsight data is not migrated — re-retain it.
- **Chunks differ in two ways.** Retain splits at 3000 tokens where
  Hindsight splits at 3000 characters, so each chunk holds more text. And
  `include.chunks` draws on the results that survived `max_tokens`, where
  Hindsight uses the ranked results from before that cut.
- Not implemented: mental-model / page / directive / webhook / file-upload
  endpoints, cancelling or deleting operations, the HTTP MCP endpoint, opinions
  via this API (a rustsight-native kind, never returned here).

HTTP API authentication
------------------------
Set `RUSTSIGHT_API_TOKEN` (or point `RUSTSIGHT_API_TOKEN_FILE` at a file
containing it) and every HTTP request must carry
`Authorization: Bearer <token>`; `/healthz` stays public for liveness
probes. Without a token the server only serves loopback addresses:
binding a non-loopback `--addr` without auth is refused unless you pass
`--allow-unauth` deliberately. The MCP stdio server needs no token — the
parent process controls the pipe, so it is not exposed.

Reranking (accuracy-critical)
-----------------------------
Recall fuses four retrieval arms with RRF and then reranks with a
cross-encoder — the paper attributes a significant share of its benchmark
lead to that reranking step. Rerank scores are normalized (calibrated [0,1]
scores pass through; logits go through a sigmoid) and modulated by
multiplicative recency and proof-count boosts (upstream's combined
scoring). With no reranker configured, real (non-mock) runs print a loud
warning and fall back to a passthrough reranker — RRF rank order seeded as
the relevance base, still modulated by the boosts (upstream's slim
deployment behavior); set `RUSTSIGHT_RERANK=off` to silence the warning
and accept the accuracy hit.

Three reranker backends:

- `local` (default choice when nothing else is set): an in-process
  bge-reranker-v2-m3 cross-encoder via the `ort` ONNX runtime. Requires a
  binary built with the opt-in feature (it adds ~50-100 MB of runtime):

      cargo build --release -p rustsight-cli --features onnx-rerank          # CPU
      cargo build --release -p rustsight-cli --features onnx-rerank-cuda    # + NVIDIA GPU

  The model is downloaded from HuggingFace on first use into
  `~/.rustsight/models/` and cached; the cache is marked complete only after
  every file landed, so later starts use it with no network access (offline
  hosts work), and `RUSTSIGHT_RERANK_MODEL_SHA256` can pin the downloaded
  model's hash. `RUSTSIGHT_RERANK_MODEL_PATH` points at
  a pre-placed model for offline hosts. Query/doc pairs are scored in
  batches (`RUSTSIGHT_RERANK_BATCH`, default 16; 16-64 is the sweet spot),
  truncated at `RUSTSIGHT_RERANK_MAX_LEN` tokens (default 512). Rerank
  scores the whole fused pool (uncapped by default, matching upstream).
- `api`: `RUSTSIGHT_RERANK_URL` (+ optional `RUSTSIGHT_RERANK_API_KEY`,
  `RUSTSIGHT_RERANK_MODEL`) pointing at a Jina/Cohere-style `/rerank`
  endpoint.
- mock mode keeps `MockRerank` (identity) — fully offline, unchanged.

### GPU (CUDA) acceleration

`RUSTSIGHT_RERANK_DEVICE=cuda` (or `auto`, the default, when the CUDA stack
is present) runs the reranker on the GPU. Two things to know:

- int8-quantized models have no CUDA kernels for their `MatMulInteger` ops,
  so the CUDA path automatically uses the fp16 export instead (tensor
  cores); CPU runs use the int8 export, which is ~3x faster than fp16 on
  CPU. The device and the default model file are paired automatically;
  explicit `RUSTSIGHT_RERANK_MODEL_REPO`/`_FILE` settings always win.
- ONNX Runtime silently falls back to CPU when CUDA libraries are missing,
  so rustsight probes the stack itself (driver + CUDA 13 runtime libs:
  `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`,
  `libcurand.so.10`) before promising CUDA. `auto` degrades to CPU
  quietly; `cuda` fails with an error naming what is missing.

  The CUDA 13 runtime libs can come from a system CUDA toolkit install or
  from NVIDIA's redistributable archives on `LD_LIBRARY_PATH`; the NVIDIA
  driver must already be installed. Roughly 1.5 GB of extra VRAM is used.

### Latency added to recall

Measured with `cargo run -p rustsight-providers --features onnx-rerank
--example rerank_latency` (bge-reranker-v2-m3, batch 16, ~60-token docs, 20
rounds; desktop with 16-core laptop CPU and RTX 3050 Ti 4 GB — expect
faster numbers on real desktop GPUs):

    config            50 docs (recall cap)      100 docs
    CPU int8          p50 1.78s   p95 2.33s     p50 3.41s   p95 3.92s
    CUDA fp16         p50 0.29s   p95 0.30s     p50 0.59s   p95 0.59s

Reranking is the dominant cost in recall at full candidate sets; the
pipeline itself (four retrieval arms + fusion) stays in the milliseconds.
Note the int8-on-CUDA trap: registering CUDA for the int8 model runs its
quantized ops on CPU with device copies in between — measured *slower* than
pure CPU, which is why device and model file are paired.

Background consolidation
------------------------
`rustsight serve` and `rustsight mcp` spawn a worker that consolidates dirty
entities automatically: it wakes on a timer (default 5s, set
RUSTSIGHT_CONSOLIDATE_INTERVAL_MS) or immediately when a retain lands
(debounced so bursts coalesce into one pass), then runs consolidation for
every bank with pending work. Retain/recall responses never wait on it, and a
crash mid-pass is safe - dirty rows persist and are re-discovered on the next
tick. Library embedders get the same behavior via
`Engine::spawn_worker(interval, max_entities)` (returns a tokio JoinHandle;
abort it to stop).

Chunking, retries, and limits
-----------------------------
Retain splits long content into token-sized chunks (default ~3000 tokens with a
200-token overlap, measured with the configured token estimator; tunable via
`EngineConfig.chunk_size_tokens` / `chunk_overlap_tokens`) and extracts each
chunk separately. Chunk-local causal fact indexes are remapped to a global
index before edges are built, and facts re-extracted from the overlap region
(normalized similarity >= 0.9) are stored once. Every provider call (LLM,
embeddings, rerank) in every pipeline — retain, recall, reflect,
consolidation, mental model — retries transient failures
(HTTP 429 / 5xx, surfaced as typed statuses) with exponential
backoff — 3 attempts, 400ms base, both configurable via `retry_max_attempts`
and `retry_base_delay_ms`; other errors (auth, bad requests) fail fast. If a
chunk's extraction output fails to parse (e.g. JSON truncated by an output
token limit), the chunk is re-asked once at half size before the error
surfaces.

Evaluation (LongMemEval-S)
--------------------------
`rustsight-eval` (RS-6) runs the paper's LongMemEval-S protocol end-to-end
against any OpenAI-compatible provider stack: retain each conversation's
sessions into a per-conversation bank (backdated via synthesized session
timestamps), consolidate, answer the 500 questions with `reflect`, and score
with a binary LLM judge (`\boxed{yes}`/`\boxed{no}`, temperature 0;
abstention questions get a dedicated unanswerable-question prompt).

    cargo run -p rustsight-eval -- download                     # ~265MB from HuggingFace
    cargo run -p rustsight-eval -- --dataset longmemeval-s --out results/
    cargo run -p rustsight-eval -- analyze --run results/       # failure clusters
    cargo run -p rustsight-eval -- compare --run results/       # vs Hindsight, per question
    scripts/fullstack-smoke                                      # 1-bank real-stack smoke first

Notes:

- Full-stack smoke before any long run: `scripts/fullstack-smoke` drives the
  committed fixture (one bank, 12 sessions, 7 questions across all five
  abilities) through retain → consolidate → reflect → judge against the
  real provider stack; it rotates its output dir so every run is fresh
  (`--mock` works offline).
- Long-run observability: per-conversation outcome lines print as they
  finish, plus a `--heartbeat-secs` (default 60) stderr heartbeat with
  phase counters. Parallelism: `--conversations` (default 2) across banks,
  `--consolidate-concurrency` (default 4) entities per consolidation pass.
  A bank whose consolidation stalls is kept partially consolidated (still
  answerable, flagged `partial` in state.json) instead of failing the run.

- Judge env (required for real runs, all three):
  `RUSTSIGHT_EVAL_JUDGE_BASE_URL`, `RUSTSIGHT_EVAL_JUDGE_API_KEY`,
  `RUSTSIGHT_EVAL_JUDGE_MODEL` — e.g. GPT-OSS-120b via OpenRouter. The
  judge never silently falls back to the extraction model.
- The engine stack reuses the production env vars (`RUSTSIGHT_LLM_*`,
  `RUSTSIGHT_EMBED_*`, `RUSTSIGHT_RERANK*`). Embeddings need any
  OpenAI-compatible `/embeddings` endpoint: OpenRouter (upstream Hindsight
  uses it as an embeddings gateway), OpenAI direct, Gemini's OpenAI-compat
  endpoint, or local Ollama/TEI/vLLM. Not Groq — no embeddings endpoint.
  Keep one embed model per run directory (mixing models breaks cosine
  similarity); see `evals/RESULTS.md` for details.
- Runs are resumable at every unit boundary (conversations, questions):
  rerun the same command and completed work is skipped. Banks live in
  `<out>/banks.db`, never the default rustsight DB.
- Staged runs: `--limit-questions 5` first (~$0.10) to calibrate extraction
  density before the full 500-question run (~$1–4 on 20B-class pricing via
  OpenRouter; see `evals/RESULTS.md` for the cost model).
- `--mock` runs the entire harness offline with mock providers and a fixed
  judge — that's what the integration tests exercise.
- Per-question traces (hypothesis, citations, all retrieved memory ids,
  judge verdict + raw output, evidence session ids, timings) land in
  `<out>/traces/`; `report.md`/`report.json` carry per-ability and overall
  accuracy against the paper's reference scores (39.0 / 83.6 / 89.0 / 91.4).
- Hindsight parity record: `compare` joins a run's traces question-by-question
  against the committed record of upstream Hindsight's own LongMemEval-S run
  (Agent Memory Benchmark: 473/500 = 94.6%, answer head Gemini-3.1-Pro,
  judge Gemini-2.5-flash-lite) in
  `crates/rustsight-eval/fixtures/hindsight_longmemeval_s.json`. It writes
  `compare.md`/`compare.json` into the run dir: agreement matrix, per-ability
  deltas, the parity gaps (Hindsight right where rustsight is not — the
  questions to target), rustsight-only wins, and shared failures. Verdicts
  come from different judges, so agreement is evidence, not ground truth.
  `scripts/hindsight-record-sync` refreshes the fixture from the benchmark
  repo and reports verdict flips.

Offline demo (no API key)
-------------------------
    rustsight --mock --db /tmp/demo.db retain --bank demo \
        --content "Last week Alice joined Google. Alice works on search infrastructure."
    rustsight --mock --db /tmp/demo.db consolidate --bank demo
    rustsight --mock --db /tmp/demo.db recall --bank demo --query "Where does Alice work?"
    rustsight --mock --db /tmp/demo.db reflect --bank demo --query "What should I know about Alice?"

Upstream prompts, sync, and parity tests
----------------------------------------
The retain extraction prompts rustsight sends are the upstream Hindsight
production prompts, vendored at a pinned release (`v1.1.0`, commit
`a5a619b5`) under `crates/rustsight-core/prompts/upstream/<tag>/` and ported
byte-for-byte into `crates/rustsight-core/src/prompts/upstream/`
(`crates/rustsight-core/prompts/upstream/PROMPTS_VERSION` records the pin).

Two scripts manage them:

    scripts/prompt-sync      # re-vendor the pinned upstream files and diff
                             # against the previous vendor to surface changes
    scripts/prompt-goldens   # regenerate texts.rs + golden fixtures by
                             # running the vendored upstream builders
                             # (needs python3 + pydantic; no network)

The parity suite (`cargo test -p rustsight-core --test
upstream_prompt_parity`) asserts the Rust builders produce output
byte-identical to fixtures generated from the upstream Python itself — 35
cases covering extraction (default/verbose/custom/labels/language modes and
the chunk user message), consolidation (system/mission/input), and reflect
(the full tools-prompt branch matrix plus final system/user prompts).
Divergences — where the engine intentionally differs (JSON-mode output
envelope, lenient parsing, per-entity consolidation passes, the reflect tool
loop's own tool set) — are recorded in
`crates/rustsight-core/prompts/upstream/DIVERGENCES.md`.

Serve
-----
    rustsight serve --addr 127.0.0.1:8888
    curl -X POST localhost:8888/banks/demo/retain -H 'content-type: application/json' \
        -d '{"content": "Alice joined Google."}'
    curl -X POST localhost:8888/banks/demo/recall -H 'content-type: application/json' \
        -d '{"query": "Where does Alice work?"}'

MCP (stdio)
-----------
    rustsight mcp --bank demo
    tools: retain(bank_id?, content), recall(bank_id?, query), reflect(bank_id?, query)
    JSON-RPC over newline-delimited stdin; malformed lines get a -32700
    parse error, lines over 10 MiB a -32600 and are discarded

Roadmap
-------
- sqlite-vec / HNSW ANN backend
- knowledge-page fs mount
- full TEMR recall parity (per-fact_type arms; see RS-15 in NEXT_STEPS.md)
- golden-pin the hand-ported final-synthesis builders via scripts/prompt-goldens
- LongMemEval-S baseline run + iteration (harness shipped as `rustsight-eval`;
  see evals/RESULTS.md)
