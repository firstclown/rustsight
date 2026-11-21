//! Semantic recall arm: brute-force scan vs the sqlite-vec KNN index (RS-7).
//!
//! `cargo bench -p rustsight-storage --bench semantic`
//!
//! Each size builds two in-memory stores holding the same random unit
//! vectors (one with the index, one without), times the recall arm's call
//! (`search_semantic`, k = 300 = reflect's MID per-arm budget, world +
//! experience kinds), and prints recall@10 of the index against the scan.
//! `RUSTSIGHT_BENCH_SIZES=10000,100000` (default) picks the sizes and
//! `RUSTSIGHT_BENCH_DIM` (default 384) the dimension.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustsight_core::types::{MemoryKind, NewMemoryRecord};
use rustsight_core::{Bank, Disposition, MemoryStore, SystemClock};
use rustsight_storage::{SqliteStore, StoreOptions};
use std::collections::HashSet;
use std::sync::Arc;

const BANK: &str = "bench";
const KINDS: [MemoryKind; 2] = [MemoryKind::World, MemoryKind::Experience];
const K: usize = 300;

/// xorshift64* — deterministic vectors without a rand dependency.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn unit(&mut self, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect()
    }
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn build(vector_index: bool, n: usize, dim: usize) -> SqliteStore {
    let store = SqliteStore::open_in_memory_with_options(
        Arc::new(SystemClock),
        &StoreOptions { vector_index },
    )
    .expect("open store");
    store
        .ensure_bank(&Bank {
            id: BANK.into(),
            mission: None,
            directives: None,
            background: None,
            disposition: Disposition::default(),
        })
        .expect("bank");
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut batch = Vec::with_capacity(1000);
    for i in 0..n {
        batch.push(NewMemoryRecord {
            kind: if i % 3 == 0 {
                MemoryKind::Experience
            } else {
                MemoryKind::World
            },
            text: format!("memory {i}"),
            confidence: None,
            occurred_start_ms: None,
            occurred_end_ms: None,
            mentioned_at_ms: i as i64,
            embedding: rng.unit(dim),
            source: None,
            context: None,
            document_id: None,
            tags: vec![],
            metadata: None,
            chunk_id: None,
        });
        if batch.len() == 1000 || i + 1 == n {
            store.insert_memories(BANK, &batch).expect("insert");
            batch.clear();
        }
    }
    store
}

fn search(store: &SqliteStore, q: &[f32], k: usize) -> Vec<String> {
    store
        .search_semantic(BANK, &[q], &KINDS, k, None)
        .expect("search")
        .pop()
        .unwrap_or_default()
        .into_iter()
        .map(|(id, _, _)| id)
        .collect()
}

fn bench_semantic(c: &mut Criterion) {
    let dim = env_list("RUSTSIGHT_BENCH_DIM", &[384])[0];
    let mut group = c.benchmark_group("semantic_arm");
    group.sample_size(10);
    for n in env_list("RUSTSIGHT_BENCH_SIZES", &[10_000, 100_000]) {
        let brute = build(false, n, dim);
        let knn = build(true, n, dim);
        assert!(
            knn.bank_vector_indexed(BANK).expect("marker"),
            "index store must answer from sqlite-vec"
        );
        // Ids differ per store (uuids); compare by text through the scan's
        // and the index's own ids, so map both to the stored text order.
        let text_of = |s: &SqliteStore, ids: &[String]| -> Vec<String> {
            ids.iter()
                .map(|id| s.get_memory(id).expect("get").expect("exists").text)
                .collect()
        };
        let mut qrng = Rng(0xD1B5_4A32_D192_ED03);
        let queries: Vec<Vec<f32>> = (0..20).map(|_| qrng.unit(dim)).collect();
        let mut hit = 0usize;
        for q in &queries {
            let truth: HashSet<String> = text_of(&brute, &search(&brute, q, 10))
                .into_iter()
                .collect();
            hit += text_of(&knn, &search(&knn, q, 10))
                .into_iter()
                .filter(|t| truth.contains(t))
                .count();
        }
        eprintln!(
            "semantic_arm n={n} dim={dim}: recall@10 index vs scan = {:.3}",
            hit as f64 / (queries.len() * 10) as f64
        );
        let q = &queries[0];
        group.bench_with_input(BenchmarkId::new("brute_force", n), &n, |b, _| {
            b.iter(|| search(&brute, q, K))
        });
        group.bench_with_input(BenchmarkId::new("sqlite_vec", n), &n, |b, _| {
            b.iter(|| search(&knn, q, K))
        });
        // Retain's shape: one batch of 20 new facts, top semantic_edge_cap
        // (5) neighbours each, across every kind.
        let batch: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();
        let all = [
            MemoryKind::World,
            MemoryKind::Experience,
            MemoryKind::Observation,
            MemoryKind::Opinion,
        ];
        for (name, store) in [
            ("brute_force_batch20", &brute),
            ("sqlite_vec_batch20", &knn),
        ] {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| {
                    store
                        .search_semantic(BANK, &batch, &all, 5, None)
                        .expect("batch")
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_semantic);
criterion_main!(benches);
