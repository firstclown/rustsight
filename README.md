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
    RUSTSIGHT_RERANK_URL      optional; any /rerank endpoint (Jina/Cohere style)
    RUSTSIGHT_DB              default ~/.rustsight/rustsight.db

Offline demo (no API key)
-------------------------
    rustsight --mock --db /tmp/demo.db retain --bank demo \
        --content "Last week Alice joined Google. Alice works on search infrastructure."
    rustsight --mock --db /tmp/demo.db consolidate --bank demo
    rustsight --mock --db /tmp/demo.db recall --bank demo --query "Where does Alice work?"
    rustsight --mock --db /tmp/demo.db reflect --bank demo --query "What should I know about Alice?"

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
- vendor upstream prompt sets behind a sync script (pinned tags)
- local ONNX cross-encoder reranker (bge-reranker-v2-m3)
- sqlite-vec / HNSW ANN backend
- LongMemEval-S evaluation harness
- knowledge-page fs mount
