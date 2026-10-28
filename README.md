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
  causal links) -> entity resolution (string + co-occurrence + temporal
  proximity scoring) -> typed graph edges (entity / temporal / semantic /
  causal) -> dirty-entity queue
- recall: four parallel retrieval arms (vector cosine, FTS5 BM25, 2-hop graph
  traversal, parsed temporal window) -> reciprocal rank fusion -> cross-encoder
  rerank -> greedy token-budget cut
- reflect: agentic tool loop (recall / read_observation / finish) shaped by the
  bank's mission, directives, and disposition traits (skepticism, literalism,
  empathy, bias strength); opinions are formed with confidence and reinforced
  over time; citations are validated against retrieved ids

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
    cargo build --release
    cargo test

Configuration (non-mock mode)
-----------------------------
    RUSTSIGHT_LLM_BASE_URL    default https://api.openai.com/v1
    RUSTSIGHT_LLM_API_KEY     required unless --mock
    RUSTSIGHT_LLM_MODEL       default gpt-4o-mini
    RUSTSIGHT_EMBED_BASE_URL  defaults to RUSTSIGHT_LLM_BASE_URL
    RUSTSIGHT_EMBED_MODEL     default text-embedding-3-small
    RUSTSIGHT_RERANK          local | api | off | auto (default auto; see below)
    RUSTSIGHT_RERANK_URL      optional; any /rerank endpoint (Jina/Cohere style)
    RUSTSIGHT_RERANK_MODEL_PATH  pre-placed ONNX model (file or dir); skips download
    RUSTSIGHT_RERANK_MODEL_SHA256  optional; hex sha256 pin — a downloaded model
                              file whose hash differs is refused
    RUSTSIGHT_RERANK_DEVICE   auto | cpu | cuda (default auto)
    RUSTSIGHT_DB              default ~/.rustsight/rustsight.db
    RUSTSIGHT_API_TOKEN       optional; bearer token required by every HTTP
                              endpoint except /healthz (see below)
    RUSTSIGHT_API_TOKEN_FILE  optional; read the token from a file instead

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
lead to that reranking step. With no reranker configured, real (non-mock)
runs print a loud warning and fall back to raw RRF ordering; set
`RUSTSIGHT_RERANK=off` to silence the warning and accept the accuracy hit.

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
  truncated at `RUSTSIGHT_RERANK_MAX_LEN` tokens (default 512). Recall caps
  rerank input at min(50, refs) candidates.
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
(normalized similarity >= 0.9) are stored once. Transient provider failures
(HTTP 429 / 5xx, surfaced as typed statuses) are retried with exponential
backoff — 3 attempts, 400ms base, both configurable via `retry_max_attempts`
and `retry_base_delay_ms`; other errors (auth, bad requests) fail fast. If a
chunk's extraction output fails to parse (e.g. JSON truncated by an output
token limit), the chunk is re-asked once at half size before the error
surfaces.

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
envelope, lenient parsing, unadopted consolidation/reflect prompts) — are
recorded in `crates/rustsight-core/prompts/upstream/DIVERGENCES.md`.

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

Roadmap
-------
- sqlite-vec / HNSW ANN backend
- LongMemEval-S evaluation harness
- knowledge-page fs mount
- adopt upstream consolidation + reflect prompts end-to-end (see
  prompts/upstream/DIVERGENCES.md)
