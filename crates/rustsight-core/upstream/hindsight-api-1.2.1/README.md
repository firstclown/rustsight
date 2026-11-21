# Vendored hindsight-api 1.2.1 sources

Verbatim copies of upstream files that rustsight ports and holds to
behavioral parity, taken from the released `hindsight-api` **1.2.1** wheel
(the version the first swap target runs). MIT License, Copyright (c) 2025
Vectorize AI, Inc. — see the repository `NOTICE`.

These features do not exist at the `v1.1.0` prompt pin
(`../../prompts/upstream/PROMPTS_VERSION`), hence the separate directory.

| File | Ported to | Parity check |
|---|---|---|
| `extensions/memory_defense.py` | `src/defense.rs` | `scripts/defense-goldens` → `tests/fixtures/memory_defense.json` → `tests/memory_defense_parity.rs` |
