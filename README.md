# Rustsight

**Long-term memory for AI agents, in one Rust binary and one SQLite file.**

Rustsight gives an agent memory that lasts across conversations. You feed it
conversations, notes, or documents. It extracts facts, links them into a
graph of entities and events, and over time distills them into observations
backed by evidence. When the agent needs context, Rustsight recalls the
relevant memories or reasons over them to answer a question.

It reimplements the architecture from
[*Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects*](https://arxiv.org/abs/2512.12818)
from scratch in Rust, with prompts held byte-for-byte to the MIT-licensed
[Hindsight](https://github.com/vectorize-io/hindsight) reference
implementation. The goal is to bring that design to places where a Python
service plus Postgres is too heavy, such as desktop apps, CLIs, and local
agents.

- **Embeddable.** Link it into your Rust application as a library, or run it
  as a standalone server.
- **One file of state.** SQLite (WAL, FTS5, vector embeddings) in a single
  database file, with no Postgres, vector database, or Python runtime.
- **Drop-in for Hindsight clients.** `rustsight serve` speaks the
  `hindsight-api` wire protocol, so the official TypeScript and Python
  clients work against it unchanged.
- **Bring your own models.** Works with any OpenAI-compatible chat and
  embeddings endpoint (OpenAI, OpenRouter, Ollama, vLLM, …). It can also run
  a local cross-encoder reranker in-process, on CPU or CUDA.
- **MCP built in.** `rustsight mcp` exposes memory as tools over stdio.

## How it works

Rustsight organizes memory into separate networks for **world facts**,
**experiences**, **observations**, and **opinions**. Three operations work
on them:

| Operation | What happens |
|---|---|
| **Retain** | An LLM extracts narrative facts from the content, with time ranges, typed entities, and causal links. Entities are resolved against what the bank already knows, and facts are joined by entity, temporal, semantic, and causal edges. |
| **Recall** | Four retrieval strategies run in parallel: vector similarity, BM25 full-text, a 2-hop graph walk, and a parsed time window. Their results are fused with reciprocal rank fusion, reranked by a cross-encoder, and trimmed to a token budget. |
| **Reflect** | An agentic tool loop searches observations and raw facts, then writes an answer with citations to the memories it used. The bank's mission, directives, and disposition traits shape the answer. |

A background worker **consolidates** new facts into observations backed by
evidence. It is never on the retain or recall path, so those stay fast.

## Benchmarks

Rustsight runs on the open
[Agent Memory Benchmark](https://github.com/vectorize-io/agent-memory-benchmark)
(AMB) through its Hindsight HTTP adapter, the same harness that produced
Hindsight's published results.

| Benchmark | Questions | Rustsight | Hindsight (AMB record) |
|---|---:|---:|---:|
| [LoCoMo](https://github.com/snap-research/locomo) (locomo10) | 1,540 | **92.7%** | 92.0% |
| [PersonaMem](https://arxiv.org/abs/2504.14225) (32k) | 589 | 77.4% | 86.6% |
| [PrecisionMemBench](https://github.com/tenurehq/precisionmembench) (single-turn) | 77 | 80.5% | 85.7% |

<details>
<summary>LoCoMo by question type, plus run setup</summary>

| LoCoMo category | Correct | Accuracy |
|---|---:|---:|
| Open-domain | 803 / 841 | 95.5% |
| Temporal | 295 / 321 | 91.9% |
| Single-hop | 258 / 282 | 91.5% |
| Multi-hop | 71 / 96 | 74.0% |

**Rustsight stack:** `openai/gpt-oss-20b` (via OpenRouter) for extraction
and consolidation, `text-embedding-3-small` embeddings, and the local
`bge-reranker-v2-m3` cross-encoder. **AMB answer and judge model:**
`z-ai/glm-5.3-flash`. LoCoMo and PersonaMem run in AMB's `rag` mode, which
uses Rustsight's recall plus the benchmark's own answer model.
PrecisionMemBench runs in `retrieval` mode, which scores the set of returned
memories directly. Rustsight's precision there was 0.80 and its recall 1.00.

**Caveats:** The Hindsight column comes from AMB's published runs, which use
different answer and judge models, so treat the comparison as indicative
rather than head-to-head. PersonaMem and PrecisionMemBench are open gaps. No
LongMemEval run has been published yet.

</details>

## Quick start

Build from source with a recent stable Rust toolchain:

```bash
git clone https://github.com/<owner>/rustsight.git
cd rustsight
cargo build --release          # binary at target/release/rustsight
```

Try it offline with mock providers. This needs no API key:

```bash
alias rs='target/release/rustsight --mock --db /tmp/demo.db'
rs retain  --bank demo --content "Last week Alice joined Google. She works on search infrastructure."
rs consolidate --bank demo
rs recall  --bank demo --query "Where does Alice work?"
rs reflect --bank demo --query "What should I know about Alice?"
```

For real results, point it at an OpenAI-compatible provider:

```bash
export RUSTSIGHT_LLM_API_KEY=sk-...                 # required
export RUSTSIGHT_LLM_MODEL=gpt-4o-mini              # default
export RUSTSIGHT_LLM_BASE_URL=https://api.openai.com/v1   # or OpenRouter, Ollama, vLLM, ...
export RUSTSIGHT_EMBED_MODEL=text-embedding-3-small # default; embeddings reuse the LLM endpoint unless set
```

### Reranking

Cross-encoder reranking accounts for much of the recall quality. Without a
reranker, Rustsight warns at startup and falls back to the fused ranking.
You have two options:

```bash
# Local, in-process bge-reranker-v2-m3 (model downloads on first use):
scripts/build-release            # CUDA build; add --cpu for CPU-only
export RUSTSIGHT_RERANK=local

# Or any Jina/Cohere-style /rerank endpoint:
export RUSTSIGHT_RERANK_URL=https://api.jina.ai/v1/rerank
export RUSTSIGHT_RERANK_API_KEY=...
```

On a laptop RTX 3050 Ti, the local reranker adds about 0.3 s to a 50-candidate
recall on CUDA, or about 1.8 s on CPU.

## Use it standalone

### HTTP server

```bash
rustsight serve --addr 127.0.0.1:8888
```

The native API is small:

```bash
curl -X POST localhost:8888/banks/demo/retain -H 'content-type: application/json' \
    -d '{"content": "Alice joined Google."}'
curl -X POST localhost:8888/banks/demo/recall -H 'content-type: application/json' \
    -d '{"query": "Where does Alice work?"}'
curl -X POST localhost:8888/banks/demo/reflect -H 'content-type: application/json' \
    -d '{"query": "What do we know about Alice?"}'
```

`GET /openapi.json` describes every endpoint, request body, and response
shape (OpenAPI 3.1; source in `crates/rustsight-server/openapi.json`).

The same server also speaks the **Hindsight API** under `/v1/default/...`.
To switch an existing Hindsight client over, point its `baseUrl` at
Rustsight:

```ts
import { HindsightClient } from "@vectorize-io/hindsight-client";

const client = new HindsightClient({ baseUrl: "http://127.0.0.1:8888" });
await client.retain("demo", "Alice joined Google.");
const { results } = await client.recall("demo", "Where does Alice work?");
```

Rustsight also accepts the `hindsight-api` command line
(`--host`, `--port`, `--log-level`) and its `HINDSIGHT_API_LLM_*`
environment variables, so a supervisor that launches Hindsight can launch
Rustsight instead. The
[compatibility matrix](docs/reference.md#hindsight-compatible-api-drop-in-for-hindsight-clients)
lists what is supported, what is ignored, and how behavior differs.

Without a token, the server binds only to loopback. Set
`RUSTSIGHT_API_TOKEN` to require `Authorization: Bearer <token>` on every
request and allow non-loopback binds.

### MCP server

```bash
rustsight mcp --bank my-agent
```

This exposes `retain`, `recall`, and `reflect` as MCP tools over stdio. For
example, in an MCP client config:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/rustsight",
      "args": ["mcp", "--bank", "my-agent"],
      "env": { "RUSTSIGHT_LLM_API_KEY": "sk-..." }
    }
  }
}
```

### CLI

`rustsight` also has `retain`, `recall`, `reflect`, `consolidate`,
`inspect`, `banks`, `bank-set` (mission, directives, and disposition),
`mental-model`, and `reindex` (rebuild the vector index for a database
written before it existed) subcommands. Run `rustsight --help` for details. The database
lives at `~/.rustsight/rustsight.db` unless you set `--db` or `RUSTSIGHT_DB`.

## Use it as a library

Add the crates to your `Cargo.toml`. They aren't on crates.io yet, so use a
git dependency:

```toml
[dependencies]
rustsight-core      = { git = "https://github.com/<owner>/rustsight" }
rustsight-storage   = { git = "https://github.com/<owner>/rustsight" }
rustsight-providers = { git = "https://github.com/<owner>/rustsight" }
tokio = { version = "1", features = ["full"] }
```

Build an `Engine` from a store, providers, and a clock, start the
consolidation worker, then call `retain`, `recall`, and `reflect`:

```rust
use std::{sync::Arc, time::Duration};

use rustsight_core::{Engine, RecallRequest, RetainRequest, SystemClock};
use rustsight_providers::{http_client, NoopRerank, OpenAiCompat, OpenAiEmbed, SimpleTokenEstimator};
use rustsight_storage::SqliteStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let clock = Arc::new(SystemClock);
    let store = Arc::new(SqliteStore::open_with_clock("memory.db", clock.clone())?);

    let client = http_client()?;
    let key = std::env::var("OPENAI_API_KEY")?;
    let base_url = "https://api.openai.com/v1".to_string();
    let llm = Arc::new(OpenAiCompat {
        client: client.clone(),
        base_url: base_url.clone(),
        api_key: key.clone(),
        model: "gpt-4o-mini".into(),
    });
    let embed = Arc::new(OpenAiEmbed {
        client,
        base_url,
        api_key: key,
        model: "text-embedding-3-small".into(),
        dim_cache: std::sync::Mutex::new(1536),
    });

    let engine = Arc::new(Engine::new(
        store,
        llm,
        embed,
        Arc::new(NoopRerank), // or ApiRerank / OrtRerank (feature "onnx-rerank")
        Arc::new(SimpleTokenEstimator),
        clock,
    ));

    // Turns new facts into observations in the background; abort to stop.
    let worker = engine.spawn_worker(Duration::from_secs(5), 50);

    engine
        .retain(RetainRequest {
            bank_id: "user-42".into(),
            content: "Alice joined Google last week. She works on search infrastructure.".into(),
            ..Default::default()
        })
        .await?;

    let recalled = engine
        .recall(RecallRequest {
            bank_id: "user-42".into(),
            query: "Where does Alice work?".into(),
            token_budget: 1000,
            top_k_per_arm: 50,
            ..Default::default()
        })
        .await?;
    for m in &recalled.memories {
        println!("{:.3}  {}", m.score, m.text);
    }

    let answer = engine.reflect("user-42", "What should I know about Alice?").await?;
    println!("{}  (cites {:?})", answer.answer, answer.citations);

    worker.abort();
    Ok(())
}
```

A runnable version is at
[`crates/rustsight-cli/examples/embed.rs`](crates/rustsight-cli/examples/embed.rs).
It falls back to mock providers when `OPENAI_API_KEY` is unset:

```bash
cargo run -p rustsight-cli --example embed
```

Some useful extension points:

- **Configuration:** `Engine::with_config(EngineConfig { .. })` covers
  chunk sizes, retries, RRF `k`, reflect rounds, consolidation timeouts, and
  more. Set `engine.retain_llm`, `engine.reflect_llm`, or
  `engine.consolidate_llm` to use a small model for extraction and a larger
  one for answers.
- **Environment-driven setup:** `rustsight_providers::ProviderEnv::from_env()`
  plus `build_providers(&env)` reads the same `RUSTSIGHT_*` variables as the
  CLI. `crates/rustsight-cli/src/main.rs::build_engine` is the reference
  recipe.
- **Your own backends:** the engine depends only on traits:
  `MemoryStore`, `LlmProvider`, `EmbedProvider`, `RerankProvider`,
  `TokenEstimator`, and `Clock`. Implement `MemoryStore` to use Postgres or
  an ANN index, or the provider traits for a non-OpenAI API.
- **Testing:** `SqliteStore::open_in_memory()`, `MockLlm::script(..)`, and
  `FixedClock` make engine tests deterministic and offline.

## Project layout

| Crate | Role |
|---|---|
| `rustsight-core` | Types, traits, prompts, and the retain/recall/reflect/consolidate pipelines |
| `rustsight-storage` | `SqliteStore`: WAL, FTS5 BM25, sqlite-vec KNN over embeddings, typed edge graph |
| `rustsight-providers` | OpenAI-compatible LLM/embeddings, API and ONNX rerankers, mocks |
| `rustsight-server` | axum HTTP API (native and Hindsight-compatible) and stdio MCP server |
| `rustsight-cli` | The `rustsight` binary |

## Documentation

- [**Reference**](docs/reference.md): every environment variable, the
  Hindsight API compatibility matrix, reranker and GPU setup, consolidation,
  chunking and retries, and prompt parity tooling.
- [`NEXT_STEPS.md`](NEXT_STEPS.md): the roadmap.
- [`AGENTS.md`](AGENTS.md): contributor guide, including architecture
  invariants and the definition of done.

## Contributing

Every change must pass all of these before it's done:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test        # fully offline: mock providers, in-memory SQLite, no API key
```

## License and attribution

MIT. See [LICENSE](LICENSE).

Rustsight implements the architecture described in *Hindsight is 20/20*
(Latimer et al., [arXiv:2512.12818](https://arxiv.org/abs/2512.12818)). It is
an independent, from-scratch Rust reimplementation inspired by
[vectorize-io/hindsight](https://github.com/vectorize-io/hindsight), not a
code port. Vectorize did not write or endorse it. The prompt sources vendored
from that project keep their MIT notices. See [NOTICE](NOTICE) for details.
