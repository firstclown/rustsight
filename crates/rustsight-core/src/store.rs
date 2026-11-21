use crate::types::*;
use crate::Result;
use std::collections::HashMap;

/// One semantic hit: item id, its kind, and cosine similarity to the query.
pub type SemanticHit = (MemoryId, MemoryKind, f32);

/// Optional id predicate applied *inside* the semantic search, before its
/// top-k cut (tag scopes, "not one of this batch's new facts").
pub type Admit<'a> = Option<&'a (dyn Fn(&str) -> bool + 'a)>;

/// Exhaustive top-k cosine search over a bank's embeddings, one result list
/// per query: kinds and `admit` filter first, then best-first by
/// similarity, ties broken by id. The reference semantics every
/// [`MemoryStore::search_semantic`] implementation must match.
pub fn brute_force_semantic(
    all: &[(MemoryId, MemoryKind, Vec<f32>)],
    queries: &[&[f32]],
    kinds: &[MemoryKind],
    k: usize,
    admit: Admit<'_>,
) -> Vec<Vec<SemanticHit>> {
    let pool: Vec<&(MemoryId, MemoryKind, Vec<f32>)> = all
        .iter()
        .filter(|(id, kind, v)| {
            !v.is_empty() && kinds.contains(kind) && admit.is_none_or(|f| f(id))
        })
        .collect();
    queries
        .iter()
        .map(|q| {
            let mut scored: Vec<SemanticHit> = pool
                .iter()
                .map(|(id, kind, v)| (id.clone(), *kind, crate::cosine(q, v)))
                .collect();
            sort_hits(&mut scored);
            scored.truncate(k);
            scored
        })
        .collect()
}

/// Best-first by similarity, ties by id — the canonical semantic order.
pub fn sort_hits(hits: &mut [SemanticHit]) {
    hits.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
}

pub trait MemoryStore: Send + Sync {
    fn ensure_bank(&self, bank: &Bank) -> Result<()>;
    fn get_bank(&self, bank_id: &str) -> Result<Option<Bank>>;
    fn update_bank(&self, bank: &Bank) -> Result<()>;
    fn list_banks(&self) -> Result<Vec<Bank>>;

    fn insert_memories(&self, bank_id: &str, items: &[NewMemoryRecord]) -> Result<Vec<Memory>>;
    fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>>;
    fn recent_memories(&self, bank_id: &str, limit: usize) -> Result<Vec<Memory>>;
    fn list_memories(&self, bank_id: &str, limit: usize) -> Result<Vec<Memory>>;
    fn all_embeddings(&self, bank_id: &str) -> Result<Vec<(MemoryId, MemoryKind, Vec<f32>)>>;
    /// Top-`k` items of `kinds` by cosine similarity, one list per query
    /// (best first, ties by id), with `admit` applied before the cut.
    /// The default scans [`MemoryStore::all_embeddings`] once for the whole
    /// batch; an indexed backend (sqlite-vec, RS-7) answers with KNN
    /// queries instead and must return the same hits.
    fn search_semantic(
        &self,
        bank_id: &str,
        queries: &[&[f32]],
        kinds: &[MemoryKind],
        k: usize,
        admit: Admit<'_>,
    ) -> Result<Vec<Vec<SemanticHit>>> {
        let all = self.all_embeddings(bank_id)?;
        Ok(brute_force_semantic(&all, queries, kinds, k, admit))
    }
    fn search_bm25(
        &self,
        bank_id: &str,
        terms: &[String],
        limit: usize,
    ) -> Result<Vec<(MemoryId, MemoryKind, f32)>>;
    fn memories_in_window(
        &self,
        bank_id: &str,
        start_ms: i64,
        end_ms: i64,
        limit: usize,
    ) -> Result<Vec<Memory>>;
    fn get_refs(&self, bank_id: &str, ids: &[MemoryId]) -> Result<Vec<MemoryRef>>;

    fn neighbors(
        &self,
        bank_id: &str,
        seeds: &[MemoryId],
        hops: usize,
        limit: usize,
    ) -> Result<Vec<MemoryId>>;

    fn create_entity(&self, e: &NewEntity) -> Result<Entity>;
    fn get_entity(&self, id: &str) -> Result<Option<Entity>>;
    fn find_by_alias(&self, bank_id: &str, alias: &str) -> Result<Option<Entity>>;
    fn entities_for_bank(&self, bank_id: &str) -> Result<Vec<Entity>>;
    fn add_alias(&self, bank_id: &str, alias: &str, entity_id: &str) -> Result<()>;
    /// Link a stored memory to resolved entities and advance each entity's
    /// `last_mentioned_ms` to `at_ms` (the fact's effective date — upstream
    /// advances its `last_seen` stat by the mention's event date).
    fn link_memory_entities(
        &self,
        memory_id: &MemoryId,
        links: &[(String, String)],
        at_ms: i64,
    ) -> Result<()>;
    fn memory_entity_ids(&self, memory_id: &MemoryId) -> Result<Vec<String>>;
    fn coentities(&self, entity_id: &str, limit: usize) -> Result<Vec<String>>;
    /// Batched [`MemoryStore::coentities`]: co-occurring entity ids for
    /// every requested entity in one pass, so a retain resolves fuzzy
    /// mentions without firing a point query (and taking the connection
    /// mutex) per candidate. Entities without co-occurrences may be absent
    /// from the map; look up with `unwrap_or_default` semantics.
    fn coentities_batch(
        &self,
        entity_ids: &[String],
        limit: usize,
    ) -> Result<HashMap<String, Vec<String>>>;
    /// Whole-bank co-occurrence snapshot for entity resolution: partner
    /// *names* per entity id plus per-name degrees, taken before the batch's
    /// new links land (upstream's resolution reads the co-occurrence table
    /// pre-commit and flushes this batch's pairs afterwards).
    fn cooccurrence_index(&self, bank_id: &str) -> Result<CooccurrenceIndex>;
    fn facts_for_entity(&self, entity_id: &str, limit: usize) -> Result<Vec<Memory>>;
    /// Batched [`MemoryStore::facts_for_entity`]: each entity's
    /// `per_entity_limit` most recently mentioned linked memories, in one
    /// pass. Within each entity the order matches `facts_for_entity`.
    fn facts_for_entities(
        &self,
        entity_ids: &[String],
        per_entity_limit: usize,
    ) -> Result<HashMap<String, Vec<Memory>>>;

    fn add_edges(&self, edges: &[Edge]) -> Result<()>;
    fn edges_for_bank(&self, bank_id: &str, limit: usize) -> Result<Vec<Edge>>;

    fn mark_entities_dirty(&self, bank_id: &str, entity_ids: &[String], reason: &str)
        -> Result<()>;
    fn list_dirty_entities(&self, bank_id: &str, limit: usize) -> Result<Vec<String>>;
    /// Dirty-entity count for a bank, without pulling the ids — progress
    /// tracking for the background worker and the eval harness's
    /// no-progress detection.
    fn count_dirty_entities(&self, bank_id: &str) -> Result<usize>;
    fn clear_dirty_entities(&self, bank_id: &str, entity_ids: &[String]) -> Result<()>;
    fn banks_with_dirty(&self) -> Result<Vec<String>>;

    fn upsert_observation(&self, o: &Observation) -> Result<()>;
    fn observations_for_entity(&self, entity_id: &str) -> Result<Vec<Observation>>;
    fn observations_for_bank(&self, bank_id: &str) -> Result<Vec<Observation>>;
    /// Remove one observation (row + FTS text) by id, scoped to its bank.
    /// Used by consolidation's `deletes` (upstream merge/supersede machine);
    /// unknown ids are a no-op.
    fn delete_observation(&self, bank_id: &str, id: &str) -> Result<()>;
    /// Attach (or clear) an embedding on an existing observation by id,
    /// without rewriting the rest of the row. Used by consolidation's
    /// post-embed backfill; unknown ids are a no-op.
    fn set_observation_embedding(&self, id: &str, embedding: Option<Vec<f32>>) -> Result<()>;

    fn upsert_opinion(&self, o: &Opinion) -> Result<()>;
    fn list_opinions(&self, bank_id: &str) -> Result<Vec<Opinion>>;

    /// Free-form per-bank settings that have no column of their own — a
    /// JSON object (`{}` when unset). Known keys: `name` (display label),
    /// `retain_mission`, `observations_mission`, `enable_observations`,
    /// `entity_labels`, `entities_allow_free_form`, and `mission_scope`
    /// (`"reflect"` when the bank's `mission` must steer reflect only, as
    /// Hindsight's `reflect_mission` does).
    fn bank_config(&self, bank_id: &str) -> Result<serde_json::Value>;
    fn set_bank_config(&self, bank_id: &str, config: &serde_json::Value) -> Result<()>;

    /// Insert or replace a document record (keeps `created_at_ms` of an
    /// existing row).
    fn upsert_document(&self, doc: &Document) -> Result<()>;
    fn get_document(&self, bank_id: &str, id: &str) -> Result<Option<Document>>;
    fn list_documents(&self, bank_id: &str, limit: usize) -> Result<Vec<Document>>;
    fn count_document_memories(&self, bank_id: &str, id: &str) -> Result<usize>;
    /// Delete every memory filed under a document, together with its FTS
    /// rows, entity links and edges, and strip the deleted ids out of
    /// observation evidence (an observation left with no evidence is
    /// removed). Returns the number of memories deleted and the entity ids
    /// whose facts or observations changed — the caller re-queues those for
    /// consolidation. The document row itself is untouched.
    fn delete_document_memories(&self, bank_id: &str, id: &str) -> Result<(usize, Vec<String>)>;
    fn delete_document(&self, bank_id: &str, id: &str) -> Result<()>;

    /// Store source chunks (insert or replace by bank + id).
    fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()>;
    /// The index the next chunk appended to a document should take.
    fn next_chunk_index(&self, bank_id: &str, document_id: &str) -> Result<usize>;
    /// Chunks by id, with their document's tags. Unknown ids are absent.
    fn get_chunks(&self, bank_id: &str, ids: &[String]) -> Result<Vec<Chunk>>;
    /// Source chunk id per memory; memories without one are absent.
    fn memory_chunk_ids(
        &self,
        bank_id: &str,
        memory_ids: &[MemoryId],
    ) -> Result<HashMap<String, String>>;

    /// Delete a bank and everything filed under it. A bank that does not
    /// exist reports zero counts.
    fn delete_bank(&self, bank_id: &str) -> Result<BankDeletion>;

    /// Tags per item id for a bank: raw memories carry their own tags,
    /// observations the union of their evidence facts' tags. Untagged items
    /// are absent from the map.
    fn tag_index(&self, bank_id: &str) -> Result<HashMap<String, Vec<String>>>;

    fn upsert_operation(&self, op: &Operation) -> Result<()>;
    fn get_operation(&self, id: &str) -> Result<Option<Operation>>;
    /// Operations still `pending`, oldest first — resumed at server start.
    fn pending_operations(&self) -> Result<Vec<Operation>>;
    /// A bank's operations, newest first, with the total matching count
    /// before `limit`/`offset`.
    fn list_operations(
        &self,
        bank_id: &str,
        status: Option<&str>,
        kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Operation>, usize)>;

    fn upsert_mental_model(&self, m: &MentalModel) -> Result<()>;
    fn list_mental_models(&self, bank_id: &str) -> Result<Vec<MentalModel>>;
    fn delete_mental_model(&self, id: &str) -> Result<()>;
}
