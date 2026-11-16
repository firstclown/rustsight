pub mod consolidation;
pub mod extraction;
pub mod prompt_utils;
pub mod reflect;
pub mod upstream;

pub use consolidation::{
    build_entity_consolidation_prompt, build_opinion_reinforcement_prompt,
    parse_consolidation_plan, parse_observations, parse_reinforcement, ConsolidationPlan,
    ParsedObservation, PlannedCreate, PlannedDelete, PlannedUpdate,
};
pub use extraction::{
    build_extraction_messages, parse_extraction, parse_extraction_with_reference, ExtractionConfig,
};
pub use reflect::{
    build_mental_model_prompt, build_opinion_extraction_prompt, build_reflect_system,
    parse_opinions, reflect_tools, verbalize_disposition, ParsedOpinion,
};
