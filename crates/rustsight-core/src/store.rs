use crate::types::*;
use crate::Result;

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
    fn link_memory_entities(&self, memory_id: &MemoryId, links: &[(String, String)]) -> Result<()>;
    fn memory_entity_ids(&self, memory_id: &MemoryId) -> Result<Vec<String>>;
    fn coentities(&self, entity_id: &str, limit: usize) -> Result<Vec<String>>;
    fn facts_for_entity(&self, entity_id: &str, limit: usize) -> Result<Vec<Memory>>;

    fn add_edges(&self, edges: &[Edge]) -> Result<()>;

    fn mark_entities_dirty(&self, bank_id: &str, entity_ids: &[String], reason: &str)
        -> Result<()>;
    fn list_dirty_entities(&self, bank_id: &str, limit: usize) -> Result<Vec<String>>;
    fn clear_dirty_entities(&self, bank_id: &str, entity_ids: &[String]) -> Result<()>;

    fn upsert_observation(&self, o: &Observation) -> Result<()>;
    fn observations_for_entity(&self, entity_id: &str) -> Result<Vec<Observation>>;
    fn observations_for_bank(&self, bank_id: &str) -> Result<Vec<Observation>>;

    fn upsert_opinion(&self, o: &Opinion) -> Result<()>;
    fn list_opinions(&self, bank_id: &str) -> Result<Vec<Opinion>>;

    fn upsert_mental_model(&self, m: &MentalModel) -> Result<()>;
    fn list_mental_models(&self, bank_id: &str) -> Result<Vec<MentalModel>>;
    fn delete_mental_model(&self, id: &str) -> Result<()>;
}
