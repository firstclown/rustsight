# Rustsight reference

The full reference: architecture details, every configuration knob, the
Hindsight-compatible API surface, reranking, and prompt parity.
Start with the [README](../README.md) for an overview.

A Hindsight-style agent memory engine written in Rust, designed as an embeddable
component for desktop applications. Implements the architecture of "Hindsight is
20/20: Building Agent Memory that Retains, Recalls, and Reflects"
(arXiv:2512.12818) with prompts informed by the upstream MIT-licensed
vectorize-io/hindsight implementation.

## Architecture
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

Storage: a single SQLite file (WAL, FTS5 for BM25, BLOB embeddings mirrored
into a sqlite-vec KNN index - see "Vector index and scale" below). Swap-in
points for other vector backends and Postgres exist via the `MemoryStore`
trait (`search_semantic`).

## Crates
- rustsight-core      types, provider traits, prompts, pipelines
- rustsight-storage   SQLite implementation of `MemoryStore`
- rustsight-providers OpenAI-compatible LLM/embeddings/rerank over HTTP, mocks
- rustsight-server    axum HTTP API + stdio MCP server
- rustsight-cli       CLI binary (`rustsight`)

## Build
    cargo build                 # debug (workspace)
    scripts/build-release       # release binary with the local ONNX reranker
                                # (CUDA; --cpu for CPU-only) — required for
                                # RUSTSIGHT_RERANK=local. A plain
                                # `cargo build --release` omits the reranker.
    cargo test

## Configuration (non-mock mode)
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
    RUSTSIGHT_VECTOR_INDEX    on | off (default on); off keeps semantic search
                              on the brute-force scan (see "Vector index")
    RUSTSIGHT_ASYNC_RETAIN_CONCURRENCY  async retains run at once through the
                              Hindsight API (default 2, range 1-64)
    RUSTSIGHT_API_TOKEN       optional; bearer token required by every HTTP
                              endpoint except /healthz (see below)
    RUSTSIGHT_API_TOKEN_FILE  optional; read the token from a file instead
    RUSTSIGHT_ENABLE_FILE_UPLOAD_API  true | false (default true); the
                              files/retain endpoint (see "File ingest")
    RUSTSIGHT_FILE_PARSER     default parser chain for uploads (default text,html)
    RUSTSIGHT_FILE_CONVERSION_MAX_BATCH_SIZE  files per upload (default 10)
    RUSTSIGHT_FILE_CONVERSION_MAX_BATCH_SIZE_MB  total upload size (default 100)
    RUSTSIGHT_WEBHOOK_URL     optional global webhook for every bank (see
                              "Webhooks"); _SECRET signs it, _EVENT_TYPES is a
                              comma list (default consolidation.completed)
    RUSTSIGHT_WEBHOOK_ALLOWED_HOSTS  hostnames / IPs / CIDRs webhooks may reach
                              despite being private or loopback
    RUSTSIGHT_WEBHOOK_EXPOSE_RESPONSE_BODY  true returns receivers' response
                              bodies from the deliveries API (default false)

Every LLM setting above also answers to its `HINDSIGHT_API_*` name
(`HINDSIGHT_API_LLM_BASE_URL` / `_API_KEY` / `_MODEL`, and the
`HINDSIGHT_API_RETAIN_LLM_*`, `_REFLECT_LLM_*`, `_CONSOLIDATION_LLM_*`
families), so a supervisor that launches `hindsight-api` with that
environment can launch rustsight unchanged; the `RUSTSIGHT_*` name wins
when both are set. Embeddings and reranking are *not* aliased — see the
compatibility notes below.

## Hindsight-compatible API (drop-in for Hindsight clients)
`rustsight serve` also speaks the upstream `hindsight-api` wire protocol, so
the official clients (`@vectorize-io/hindsight-client`,
`hindsight-client` for Python) work against it unchanged — point their
`baseUrl` at the rustsight server. Shapes follow hindsight-api 1.2.x; errors
use FastAPI's `{"detail": "..."}` body. The native API (`/banks/...`) is
untouched and served side by side.

    GET    /health, /version                               (public)
    GET    /openapi.json                                   (public; see below)
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
    POST   /v1/default/banks/{bank_id}/files/retain        multipart upload, async
    GET    /v1/default/banks/{bank_id}/attachments/{id}    a file's original bytes
    POST   /v1/default/banks/{bank_id}/webhooks            register (GET lists)
    PATCH  /v1/default/banks/{bank_id}/webhooks/{id}       (DELETE too)
    GET    /v1/default/banks/{bank_id}/webhooks/{id}/deliveries

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
  `enable_observations`, `entity_labels`, `entities_allow_free_form`,
  `memory_defense` (see [Memory Defense](#memory-defense)). A bank
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
- **Webhook event types are validated.** Registering an event type other
  than the three below is a 422; Hindsight accepts it and never fires it.
- **File parsers are `text` and `html`.** Hindsight's `markitdown` / `iris`
  are not built in; naming them is the same 400 Hindsight gives for a
  parser the server does not enable. PDF and Office files fail their
  operation until a parser for them is registered.
- Not implemented: mental-model / page / directive
  endpoints, cancelling or deleting operations, the HTTP MCP endpoint, opinions
  via this API (a rustsight-native kind, never returned here).

## OpenAPI description
`GET /openapi.json` (public, no token) returns one OpenAPI 3.1 document for
both surfaces:

- **Native API** (`/healthz`, `/banks/...`): fully described — request
  bodies (`Native*Request` schemas), response shapes, error codes, and the
  bearer-token security scheme. The source is the hand-written
  `crates/rustsight-server/openapi.json`; tests in `http.rs` fail if it
  drifts from the router (paths and methods), from the request-body
  structs (field names and which are required), or from the core types'
  serialized fields and enum values.
- **Hindsight-compatible API** (`/v1/default/...`): paths only, plus the
  `RecallRequest` field list. A recall field is listed only when rustsight
  acts on it, so clients that probe the document for a feature get an
  honest answer. `info.version` stays the Hindsight API version (`1.1.0`).

Native request bodies refuse fields they do not know with a 422 (for
example ``unknown field `kinds` ``) instead of ignoring them, so the schemas
list every field the server acts on and nothing else. Adding a native route
or body field means adding it to `openapi.json` too; the tests point at
whatever is missing.

## HTTP API authentication
Set `RUSTSIGHT_API_TOKEN` (or point `RUSTSIGHT_API_TOKEN_FILE` at a file
containing it) and every HTTP request must carry
`Authorization: Bearer <token>`; `/healthz` stays public for liveness
probes. Without a token the server only serves loopback addresses:
binding a non-loopback `--addr` without auth is refused unless you pass
`--allow-unauth` deliberately. The MCP stdio server needs no token — the
parent process controls the pipe, so it is not exposed.

## Reranking (accuracy-critical)
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

## Background consolidation
`rustsight serve` and `rustsight mcp` spawn a worker that consolidates dirty
entities automatically: it wakes on a timer (default 5s, set
RUSTSIGHT_CONSOLIDATE_INTERVAL_MS) or immediately when a retain lands
(debounced so bursts coalesce into one pass), then runs consolidation for
every bank with pending work. Retain/recall responses never wait on it, and a
crash mid-pass is safe - dirty rows persist and are re-discovered on the next
tick. Library embedders get the same behavior via
`Engine::spawn_worker(interval, max_entities)` (returns a tokio JoinHandle;
abort it to stop).

## Vector index and scale
The semantic recall arm, and retain's semantic-edge neighbour search, go
through `MemoryStore::search_semantic`. `SqliteStore` answers it from a
[sqlite-vec](https://github.com/asg017/sqlite-vec) `vec0` index (0.1.9,
compiled into the binary, no loadable extension to install; storage cargo
feature `sqlite-vec`, on by default):

- one `vec_f32_<dim>` virtual table per embedding width, partitioned by
  bank, memory kind as a filter column. Memories, observations, and
  opinions are mirrored write-through, in the same transaction as the row.
- tag scopes still filter before the top-k cut: the KNN fetch widens (x4,
  up to vec0's k limit of 4096) until enough admitted rows come back, then
  falls back to the scan.
- a bank uses the index only when it is marked indexed: banks created by an
  index-maintaining store are marked from the start. Banks in a database
  written before RS-7, or written by a build or run without the index
  (`RUSTSIGHT_VECTOR_INDEX=off`, `--no-default-features`, extension failed
  to load), stay on the scan (a one-time stderr note says so) until you run

      rustsight reindex --bank <id>     # or omit --bank for every bank

  `reindex` needs no API key. It rebuilds that bank's index from the stored
  embeddings in one transaction.
- if the extension cannot initialize, the store logs it and runs every
  semantic query as a brute-force scan; nothing else changes.
- vectors wider than 8192 dimensions cannot go in vec0; a bank holding one
  stays on the scan.

sqlite-vec 0.1.x is an **exact** KNN (a flat scan in C), not an approximate
one: results match the scan (recall@10 = 1.000 in the bench), only faster
and without loading every embedding into memory per query. Measured with
`cargo bench -p rustsight-storage --bench semantic` (16-core laptop CPU,
in-memory DB, k = 300, world + experience kinds):

                              brute force    sqlite-vec
    10k vectors  x 384-d      15.9 ms        8.2 ms
    100k vectors x 384-d      131.9 ms       75.7 ms
    10k vectors  x 1536-d     47.9 ms        23.6 ms
    50k vectors  x 1536-d     247.3 ms       115.6 ms
    retain batch, 20 queries,
    50k x 1536-d              1.85 s         2.03 s

Single queries (the recall arm) are about 2x faster. A retain batch is about
break-even: the scan decodes the bank once and reuses it for every new
fact, while the index runs one KNN per fact but never holds the bank's
embeddings in memory (~300 MB at 50k x 1536-d). Rough guidance: up to
~50k memories per bank, either path is fine; at 100k+ x 1536-d, expect
~0.2 s per semantic arm with the index. Going well past that needs a true
ANN index (HNSW, or a newer sqlite-vec with ANN support) behind the same
`search_semantic` seam. Only same-width vectors are compared under the
index; the scan's cosine truncates mismatched widths, so a bank that mixes
embedding models should be re-embedded either way.

## Chunking, retries, and limits
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

## File ingest
`POST /v1/default/banks/{id}/files/retain` takes Hindsight's multipart
upload: one or more `files` parts plus a `request` JSON part with an
optional `parser` (a name or an ordered fallback list) and
`files_metadata` (one entry per file: `document_id`, `context`, `tags`,
`metadata`, `timestamp`, `parser`). It is always asynchronous and answers
`{"operation_ids": [...]}`, one `file_convert_retain` operation per file.

Each file's bytes are stored with its document (generated as
`file_<uuid>` when not given) before the response returns, so a restart
resumes the work. The operation converts the bytes with the parser chain
and retains the text like a JSON retain item — Memory Defense, tags and
timestamps apply — and every resulting fact carries `attachment_id` and
`filename` in its metadata. `GET /v1/default/banks/{id}/attachments/{attachment_id}`
serves the original bytes with the uploaded Content-Type.

- Parsers: `text` (UTF-8 text, markdown, CSV, JSON, YAML, XML, source code;
  declines HTML and binaries) and `html` (visible text: scripts, styles and
  comments dropped, entities decoded). A chain tries each in order until
  one succeeds; the default chain is `RUSTSIGHT_FILE_PARSER` (`text,html`).
  Embedders add converters with `Engine::with_parsers` and a
  `rustsight_core::files::FileParser`.
- Limits: `RUSTSIGHT_FILE_CONVERSION_MAX_BATCH_SIZE` files (10) and
  `_MB` total megabytes (100) per request; more is a 400, as are an
  unavailable parser, a `files_metadata` count that does not match the files,
  and per-file fields placed at the top level of `request`.
- Attachment ids are the first 12 hex characters of the file's sha256.
  Re-uploading into a document replaces its attachment once the new retain
  succeeds; deleting the document or the bank deletes its attachments.
- Hindsight converts in one operation and retains in a second; here one
  operation does both, so waiting on the returned id covers the retain.
- Locally: `rustsight retain --bank B --file notes.md [--parser text]
  [--document-id D]` runs the same conversion in-process.

## Webhooks
Rustsight reports three lifecycle events, with Hindsight's names, statuses
and payloads:

| Event | Fires | `status` | `data` |
|---|---|---|---|
| `retain.completed` | once per document, after its memories are stored | `completed` | `document_id`, `tags`, `memory_unit_count` (0 = extraction found nothing) |
| `consolidation.completed` | after each consolidation pass that had queued work | `completed` / `failed` | `observations_created` / `_updated` / `_deleted`, or `error_message` |
| `memory_defense.triggered` | when [Memory Defense](#memory-defense) redacts or blocks an item | `redact` / `block` | `action`, `detector`, `document_id`, `matched_types`, `message`, `hits` (fingerprints only) |

The body is `{"event_id", "event", "bank_id", "operation_id", "status",
"timestamp", "data"}`; `event_id` is shared by every delivery of one event
and stable across retries, so receivers dedupe on it. All items of one
retain request share an `operation_id`.

Register webhooks per bank through the Hindsight API
(`POST /v1/default/banks/{id}/webhooks` with `url`, `secret`, `event_types`,
`enabled`, `http_config: {method, timeout_seconds, headers, params}`), or
set one global webhook with `RUSTSIGHT_WEBHOOK_URL`. Secrets are never
returned. Delivery follows Hindsight:

- Headers `X-Hindsight-Event`, `X-Hindsight-Event-Id`,
  `X-Hindsight-Delivery-Id`; with a secret, `X-Hindsight-Signature` and
  `X-Hub-Signature-256` (`sha256=<HMAC-SHA256 of the body>`) plus
  `X-Hindsight-Signature-V2` (`t=<unix secs>,v1=<HMAC of "<t>.<body>">`,
  for replay windows). Custom headers cannot override the Hindsight ones.
- A non-2xx answer or a transport failure retries after 5s, 5m, 30m, 2h and
  5h (six attempts in all), then the delivery is `failed`. Deliveries are
  stored when the event fires, so a restart resumes them. They are not
  operations: waiting for zero pending operations never waits on a webhook.
- **SSRF guard.** Loopback, private, link-local (incl. the cloud metadata
  address), CGNAT, reserved and multicast destinations are refused unless
  `RUSTSIGHT_WEBHOOK_ALLOWED_HOSTS` lists the host or a covering IP/CIDR —
  IP literals at registration, every resolved address at delivery (the
  connection is pinned to the checked addresses; redirects are not
  followed). For a receiver on the same machine, set
  `RUSTSIGHT_WEBHOOK_ALLOWED_HOSTS=127.0.0.1`.
- `GET .../webhooks/{id}/deliveries` lists attempts, status codes and
  errors, newest first (`limit`, `cursor`).

Embedders get the same events without HTTP: `Engine::with_event_sink`
(or `set_event_sink` on a shared engine) takes any
`rustsight_core::events::EventSink`.

## Memory Defense
Opt-in, per-bank screening of everything retained for secrets and PII — a
port of upstream hindsight-api 1.2.1's regex Memory Defense. The catalog is
upstream's 46 patterns verbatim: LLM-provider keys (Anthropic, OpenAI,
Google, xAI, Groq, Hugging Face, …), cloud and CI credentials (AWS, GitHub,
GitLab, npm, PyPI, DigitalOcean), payment and chat tokens (Stripe, Square,
Slack, Twilio, SendGrid, Discord, Telegram, …), database URLs with embedded
passwords, PEM private-key headers, JWTs, Luhn-checked card numbers and US
SSNs. Token boundaries are ASCII-only, so a key embedded in Chinese or
Japanese prose still matches. `tests/memory_defense_parity.rs` holds the
Rust port to upstream's own output (`scripts/defense-goldens` regenerates the
fixture from the vendored source).

Configure it with the bank setting `memory_defense`:

```json
{"enabled": true, "rules": [{"on": "sensitive_data", "action": "redact"}]}
```

- `redact` replaces each match with `[REDACTED:<label>]` (e.g.
  `[REDACTED:github_token]`) **before** chunking and extraction, so the
  document body, chunks, facts and embeddings never see the secret. The
  retain response's `defense` field lists the matched labels with
  fingerprinted previews (`ghp_...AAAA`), never the raw value.
- `block` refuses the item: native retain answers 422; a Hindsight-API batch
  drops blocked items and retains the rest, or answers 422 with
  `{"detail": {"violations": [{"index", "detector", "message"}]}}` when every
  item is blocked.
- `allow`, `enabled: false`, or no setting: no screening. Rules naming
  another detector are accepted and ignored. A malformed policy is refused
  when written, and a retain under one fails rather than skipping screening.

Set it with `PATCH /v1/default/banks/{id}/config`
(`{"updates": {"memory_defense": {...}}}`) or
`rustsight bank-set --bank b --memory-defense '<json>'` (`off` removes it).
The patterns target high-confidence prefixes; context-dependent secrets
(a bare 40-character string next to the word "cohere") are out of scope, as
upstream.

## Offline demo (no API key)
    rustsight --mock --db /tmp/demo.db retain --bank demo \
        --content "Last week Alice joined Google. Alice works on search infrastructure."
    rustsight --mock --db /tmp/demo.db consolidate --bank demo
    rustsight --mock --db /tmp/demo.db recall --bank demo --query "Where does Alice work?"
    rustsight --mock --db /tmp/demo.db reflect --bank demo --query "What should I know about Alice?"

## Upstream prompts, sync, and parity tests
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

## Serve
    rustsight serve --addr 127.0.0.1:8888
    curl -X POST localhost:8888/banks/demo/retain -H 'content-type: application/json' \
        -d '{"content": "Alice joined Google."}'
    curl -X POST localhost:8888/banks/demo/recall -H 'content-type: application/json' \
        -d '{"query": "Where does Alice work?"}'

## MCP (stdio)
    rustsight mcp --bank demo
    tools: retain(bank_id?, content), recall(bank_id?, query), reflect(bank_id?, query)
    JSON-RPC over newline-delimited stdin; malformed lines get a -32700
    parse error, lines over 10 MiB a -32600 and are discarded

## Roadmap
- true ANN (HNSW) vector backend beyond ~100k memories per bank
- knowledge-page fs mount
- full TEMR recall parity (per-fact_type arms; see RS-15 in NEXT_STEPS.md)
- golden-pin the hand-ported final-synthesis builders via scripts/prompt-goldens
