//! Upstream prompt text constants, generated verbatim from the
//! vendored Python sources at prompts/upstream/v1.1.0/engine/ by
//! scripts/prompt-goldens. Do not edit by hand; regenerate instead.
//! MIT Licensed upstream, Copyright 2025 Vectorize AI, Inc.
//!
//! Some constants feed builders not yet ported; they stay so the file
//! matches the full upstream constant set.
#![allow(dead_code)]

/// Upstream `retain` prompt text block.
pub const EXTRACTION_DEFAULT_LANGUAGE_RULE: &str = r#"LANGUAGE: Write every fact in the same language and script as the input text. Never translate. Names, identifiers, code, and quoted text stay verbatim."#;

/// Upstream `retain` prompt text block.
pub const EXTRACTION_CONCISE_TEMPLATE: &str = r#"Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

{language_section}{retain_mission_section}══════════════════════════════════════════════════════════════════════════
SELECTIVITY - CRITICAL (Reduces 90% of unnecessary output)
══════════════════════════════════════════════════════════════════════════

ONLY extract facts that are:
✅ Personal info: names, relationships, roles, background
✅ Preferences: likes, dislikes, habits, interests (e.g., "Alice likes coffee")
✅ Significant events: milestones, decisions, achievements, changes
✅ Plans/goals: future intentions, deadlines, commitments
✅ Expertise: skills, knowledge, certifications, experience
✅ Important context: projects, problems, constraints
✅ Sensory/emotional details: feelings, sensations, perceptions that provide context
✅ Observations: descriptions of people, places, things with specific details

DO NOT extract:
❌ Generic greetings: "how are you", "hello", pleasantries without substance
❌ Pure filler: "thanks", "sounds good", "ok", "got it", "sure"
❌ Process chatter: "let me check", "one moment", "I'll look into it"
❌ Repeated info: if already stated, don't extract again

CONSOLIDATE related statements into ONE fact when possible.

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. "what": Core fact - concise but complete (1-2 sentences max)
2. "when": Temporal info if mentioned. "N/A" if none. Use day name when known.
3. "where": Location if relevant. "N/A" if none.
4. "who": People involved with relationships. "N/A" if just general info.
5. "why": Context/significance ONLY if important. "N/A" if obvious.

CONCISENESS: Capture the essence, not every word. One good sentence beats three mediocre ones.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION
══════════════════════════════════════════════════════════════════════════

Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

══════════════════════════════════════════════════════════════════════════
CLASSIFICATION
══════════════════════════════════════════════════════════════════════════

fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": Objective/external facts, including the user's preferences, rules, corrections, constraints, plans, traits, or context. These stay "world" even when the user states them during an assistant interaction (e.g., "User prefers browser_navigate over web_search", "User corrected the project deadline").
- "assistant": Actions, experiences, or observations the assistant/agent actually performed (e.g., "I changed X", "I discovered Y", "I debugged Z"). Use this for the assistant/agent doing, trying, learning, deciding, recommending, or responding — not merely for user facts mentioned in conversation.

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- CRITICAL: Convert ALL relative temporal expressions to absolute dates in the fact text itself.
  "yesterday" → write the resolved date (e.g. "on November 12, 2024"), NOT the word "yesterday"
  "last night", "this morning", "today", "tonight" → convert to the resolved absolute date
- For events: set occurred_start AND occurred_end (same for point events)
- Coarse dates (only a year, or only a month, is stated): span the WHOLE period —
  "in 2015" → 2015-01-01 to 2015-12-31, "in March 2026" → 2026-03-01 to 2026-03-31.
  Never collapse to the period's first day or to the Event Date, current year included.
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

ALWAYS return "entities" as an array of plain strings — never objects, never null.
Correct: entities=["Alice", "Kubernetes", "CKA"]
Wrong:   entities as an array of objects with a "text" key ← never use this form
Use an empty array [] only when the fact truly names nothing.

Include: people names, organizations, places, key objects, abstract concepts (career, friendship, etc.)
Always include "user" when fact is about the user.

══════════════════════════════════════════════════════════════════════════
EXAMPLES (shown in English for illustration; for non-English input, ALL output values MUST be in the input language)
══════════════════════════════════════════════════════════════════════════

The examples below demonstrate output format and selectivity only. Never emit
their facts, entities, or dates unless those details also appear in the actual
input text being processed.

Example 1 - Selective extraction (Event Date: June 10, 2024):
Input: "Hey! How's it going? Good morning! So I'm planning my wedding - want a small outdoor ceremony. Just got back from Emily's wedding, she married Sarah at a rooftop garden. It was nice weather. I grabbed a coffee on the way."

Output: ONLY 2 facts (skip greetings, weather, coffee):
1. what="User planning wedding, wants small outdoor ceremony", who="user", why="N/A", entities=["user", "wedding"]
2. what="Emily married Sarah at rooftop garden", who="Emily (user's friend), Sarah", occurred_start="2024-06-09", entities=["Emily", "Sarah", "wedding"]

Example 2 - Professional context:
Input: "Alice has 5 years of Kubernetes experience and holds CKA certification. She's been leading the infrastructure team since March. By the way, she prefers dark roast coffee."

Output: ONLY 2 facts (skip coffee preference - too trivial):
1. what="Alice has 5 years Kubernetes experience, CKA certified", who="Alice", entities=["Alice", "Kubernetes", "CKA"]
2. what="Alice leads infrastructure team since March", who="Alice", entities=["Alice", "infrastructure"]

══════════════════════════════════════════════════════════════════════════
QUALITY OVER QUANTITY
══════════════════════════════════════════════════════════════════════════

Ask: "Would this be useful to recall in 6 months?" If no, skip it.

IMPORTANT: Sensory/emotional details and observations that provide meaningful context
about experiences ARE important to remember, even if they seem small (e.g., how food
tasted, how someone looked, how loud music was). Extract these if they characterize
an experience or person."#;

/// Upstream `retain` prompt text block.
pub const EXTRACTION_CUSTOM_TEMPLATE: &str = r#"Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

{language_section}{retain_mission_section}{custom_instructions}

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. "what": Core fact - concise but complete (1-2 sentences max)
2. "when": Temporal info if mentioned. "N/A" if none. Use day name when known.
3. "where": Location if relevant. "N/A" if none.
4. "who": People involved with relationships. "N/A" if just general info.
5. "why": Context/significance ONLY if important. "N/A" if obvious.

CONCISENESS: Capture the essence, not every word. One good sentence beats three mediocre ones.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION
══════════════════════════════════════════════════════════════════════════

Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

══════════════════════════════════════════════════════════════════════════
CLASSIFICATION
══════════════════════════════════════════════════════════════════════════

fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": Objective/external facts, including the user's preferences, rules, corrections, constraints, plans, traits, or context. These stay "world" even when the user states them during an assistant interaction (e.g., "User prefers browser_navigate over web_search", "User corrected the project deadline").
- "assistant": Actions, experiences, or observations the assistant/agent actually performed (e.g., "I changed X", "I discovered Y", "I debugged Z"). Use this for the assistant/agent doing, trying, learning, deciding, recommending, or responding — not merely for user facts mentioned in conversation.

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- CRITICAL: Convert ALL relative temporal expressions to absolute dates in the fact text itself.
  "yesterday" → write the resolved date (e.g. "on November 12, 2024"), NOT the word "yesterday"
  "last night", "this morning", "today", "tonight" → convert to the resolved absolute date
- For events: set occurred_start AND occurred_end (same for point events)
- Coarse dates (only a year, or only a month, is stated): span the WHOLE period —
  "in 2015" → 2015-01-01 to 2015-12-31, "in March 2026" → 2026-03-01 to 2026-03-31.
  Never collapse to the period's first day or to the Event Date, current year included.
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

ALWAYS return "entities" as an array of plain strings — never objects, never null.
Correct: entities=["Alice", "Kubernetes", "CKA"]
Wrong:   entities as an array of objects with a "text" key ← never use this form
Use an empty array [] only when the fact truly names nothing.

Include: people names, organizations, places, key objects, abstract concepts (career, friendship, etc.)
Always include "user" when fact is about the user."#;

/// Upstream `retain` prompt text block.
pub const EXTRACTION_VERBOSE_TEMPLATE: &str = r#"Extract facts from text into structured format with FIVE required dimensions - BE EXTREMELY DETAILED.

{language_section}{retain_mission_section}══════════════════════════════════════════════════════════════════════════
FACT FORMAT - ALL FIVE DIMENSIONS REQUIRED - MAXIMUM VERBOSITY
══════════════════════════════════════════════════════════════════════════

For EACH fact, CAPTURE ALL DETAILS - NEVER SUMMARIZE OR OMIT:

1. **what**: WHAT happened - COMPLETE description with ALL specifics (objects, actions, quantities, details)
2. **when**: WHEN it happened - ALWAYS include temporal info with DAY OF WEEK (e.g., "Monday, June 10, 2024")
   - Always include the day name: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
   - Format: "day_name, month day, year" (e.g., "Saturday, June 9, 2024")
3. **where**: WHERE it happened or is about - SPECIFIC locations, places, areas, regions (if applicable)
4. **who**: WHO is involved - ALL people/entities with FULL relationships and background
5. **why**: WHY it matters - ALL emotions, preferences, motivations, significance, nuance
   - For assistant facts: MUST include what the user asked/requested that triggered this!

Plus: fact_type, fact_kind, entities, occurred_start/end (for structured dates), where (structured location)

VERBOSITY REQUIREMENT: Include EVERY detail mentioned. More detail is ALWAYS better than less.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION (CRITICAL)
══════════════════════════════════════════════════════════════════════════

When text uses BOTH a generic relation AND a name for the same person → LINK THEM!

Example input: "I went to my college roommate's wedding last June. Emily finally married Sarah after 5 years together."

CORRECT output:
- what: "Emily got married to Sarah at a rooftop garden ceremony"
- when: "Saturday, June 8, 2024, after dating for 5 years"
- where: "downtown San Francisco, at a rooftop garden venue"
- who: "Emily (user's college roommate), Sarah (Emily's partner of 5 years)"
- why: "User found it romantic and beautiful, dreams of similar outdoor ceremony"
- where (structured): "San Francisco"

WRONG output:
- what: "User's roommate got married" ← LOSES THE NAME!
- who: "the roommate" ← WRONG - use the actual name!
- where: (missing) ← WRONG - include the location!

══════════════════════════════════════════════════════════════════════════
FACT_KIND CLASSIFICATION (CRITICAL FOR TEMPORAL HANDLING)
══════════════════════════════════════════════════════════════════════════

⚠️ MUST set fact_kind correctly - this determines whether occurred_start/end are set!

fact_kind="event" - USE FOR:
- Actions that happened at a specific time: "went to", "attended", "visited", "bought", "made"
- Past events: "yesterday I...", "last week...", "in March 2020..."
- Future plans with dates: "will go to", "scheduled for"
- Examples: "I went to a pottery workshop" → event
           "Alice visited Paris in February" → event
           "I bought a new car yesterday" → event
           "The user graduated from MIT in March 2020" → event

fact_kind="conversation" - USE FOR:
- Ongoing states: "works as", "lives in", "is married to"
- Preferences: "loves", "prefers", "enjoys"
- Traits/abilities: "speaks fluent French", "knows Python"
- Examples: "I love Italian food" → conversation
           "Alice works at Google" → conversation
           "I prefer outdoor dining" → conversation

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING (CRITICAL - USE EVENT DATE AS REFERENCE)
══════════════════════════════════════════════════════════════════════════

⚠️ IMPORTANT: Use the "Event Date" provided in the input as your reference point!
All relative dates ("yesterday", "last week", "recently") must be resolved relative to the Event Date, NOT today's date.

For EVENTS (fact_kind="event") - MUST SET BOTH occurred_start AND occurred_end:
- Convert relative dates → absolute using Event Date as reference
- If Event Date is "Saturday, March 15, 2020", then "yesterday" = Friday, March 14, 2020
- Dates mentioned in text (e.g., "in March 2020") should use THAT year, not current year
- CRITICAL: If the content mentions an absolute date (e.g., "March 15, 2024", "2024-03-15"), you MUST extract it and set occurred_start in ISO format
- Always include the day name (Monday, Tuesday, etc.) in the 'when' field
- Set occurred_start AND occurred_end to WHEN IT HAPPENED (not when mentioned)
- For single-day/point events: set occurred_end = occurred_start (same timestamp)
- COARSE DATES (the text states only a year, or only a month): set occurred_start and
  occurred_end to the FULL SPAN of that period, so the range says how precisely the date
  is actually known:
    "in 2015"        → occurred_start="2015-01-01T00:00:00", occurred_end="2015-12-31T23:59:59"
    "in March 2026"  → occurred_start="2026-03-01T00:00:00", occurred_end="2026-03-31T23:59:59"
  Do NOT collapse a coarse date to the first day of the period, and do NOT resolve it to the
  Event Date — even when the period is the current year. A year-only date is not a precise
  date; recording it as one makes the memory claim a day it never stated.
- OPEN-ENDED events (still ongoing, "since forever", no stated end): set only the bound the
  text gives. NEVER write a placeholder year such as 0001 or 9999 for an unknown bound.

For CONVERSATIONS (fact_kind="conversation"):
- General info, preferences, ongoing states → NO occurred dates
- Examples: "loves coffee", "works as engineer"

══════════════════════════════════════════════════════════════════════════
FACT TYPE
══════════════════════════════════════════════════════════════════════════

- **world**: User's life, preferences, rules, corrections, constraints, other people, and events (facts that would exist without this conversation)
- **assistant**: Actions or experiences the assistant/agent actually performed while helping the user (requests, recommendations, help)
  ⚠️ CRITICAL for assistant facts: ALWAYS capture the user's request/question in the fact!
  Include: what the user asked, what problem they wanted solved, what context they provided

══════════════════════════════════════════════════════════════════════════
ENTITIES - EXTRACT EVERYTHING
══════════════════════════════════════════════════════════════════════════

Extract ALL of the following from the fact:
- People names (Emily, Alice, Dr. Smith)
- Organizations (Google, MIT, local coffee shop)
- Places (San Francisco, Brooklyn, Paris)
- Significant objects mentioned (coffee maker, new car, wedding dress)
- Abstract concepts/themes (friendship, career growth, loss, celebration)

ALWAYS include "user" when fact is about the user.
Extract anything that could help link related facts together."#;

/// Upstream `retain` prompt text block.
pub const EXTRACTION_VERBATIM_TEMPLATE: &str = r#"Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

{language_section}{retain_mission_section}══════════════════════════════════════════════════════════════════════════
VERBATIM MODE — Extract metadata only
══════════════════════════════════════════════════════════════════════════

The original text will be stored as-is in code. Your ONLY job is to extract metadata.

RULES:
- Produce EXACTLY ONE entry per input chunk.
- DO NOT include a "what" field — it is not part of the output schema.
- Extract all entities (people, places, organizations, objects, concepts).
- Extract temporal information (occurred_start, occurred_end, fact_kind, when).
- Extract location (where) and people (who).
- fact_type: use "world" for user preferences, rules, corrections, constraints, traits, and other objective facts, even when stated during an assistant interaction. Use "assistant" only for actions or experiences the assistant/agent actually performed.

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. "what": Core fact - concise but complete (1-2 sentences max)
2. "when": Temporal info if mentioned. "N/A" if none. Use day name when known.
3. "where": Location if relevant. "N/A" if none.
4. "who": People involved with relationships. "N/A" if just general info.
5. "why": Context/significance ONLY if important. "N/A" if obvious.

CONCISENESS: Capture the essence, not every word. One good sentence beats three mediocre ones.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION
══════════════════════════════════════════════════════════════════════════

Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

══════════════════════════════════════════════════════════════════════════
CLASSIFICATION
══════════════════════════════════════════════════════════════════════════

fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": Objective/external facts, including the user's preferences, rules, corrections, constraints, plans, traits, or context. These stay "world" even when the user states them during an assistant interaction (e.g., "User prefers browser_navigate over web_search", "User corrected the project deadline").
- "assistant": Actions, experiences, or observations the assistant/agent actually performed (e.g., "I changed X", "I discovered Y", "I debugged Z"). Use this for the assistant/agent doing, trying, learning, deciding, recommending, or responding — not merely for user facts mentioned in conversation.

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- CRITICAL: Convert ALL relative temporal expressions to absolute dates in the fact text itself.
  "yesterday" → write the resolved date (e.g. "on November 12, 2024"), NOT the word "yesterday"
  "last night", "this morning", "today", "tonight" → convert to the resolved absolute date
- For events: set occurred_start AND occurred_end (same for point events)
- Coarse dates (only a year, or only a month, is stated): span the WHOLE period —
  "in 2015" → 2015-01-01 to 2015-12-31, "in March 2026" → 2026-03-01 to 2026-03-31.
  Never collapse to the period's first day or to the Event Date, current year included.
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

ALWAYS return "entities" as an array of plain strings — never objects, never null.
Correct: entities=["Alice", "Kubernetes", "CKA"]
Wrong:   entities as an array of objects with a "text" key ← never use this form
Use an empty array [] only when the fact truly names nothing.

Include: people names, organizations, places, key objects, abstract concepts (career, friendship, etc.)
Always include "user" when fact is about the user."#;

/// Upstream `retain` prompt text block.
pub const EXTRACTION_CAUSAL_SECTION: &str = r#"

══════════════════════════════════════════════════════════════════════════
CAUSAL RELATIONSHIPS
══════════════════════════════════════════════════════════════════════════

Link facts with causal_relations (max 2 per fact). target_index must be < this fact's index.
Type: "caused_by" (this fact was caused by the target fact)

Example: "Lost job → couldn't pay rent → moved apartment"
- Fact 0: Lost job, causal_relations: null
- Fact 1: Couldn't pay rent, causal_relations: [{target_index: 0, relation_type: "caused_by"}]
- Fact 2: Moved apartment, causal_relations: [{target_index: 1, relation_type: "caused_by"}]"#;

/// Upstream `retain` prompt text block.
pub const EXTRACTION_OPTIONAL_DIMENSIONS_SECTION: &str = r#"

══════════════════════════════════════════════════════════════════════════
OPTIONAL FIELDS
══════════════════════════════════════════════════════════════════════════

"when", "where", "who" and "why" may be null. Write null — not "N/A" — when the
text states no value for the fact you are writing, and never carry over a value
the text states about a different subject.
"#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_DEFAULT_MISSION: &str = r#"Track anything notable in the new facts — names, numbers, dates, places, events, decisions, claims, relationships, and recurring patterns."#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_MISSION_PRIORITY_NOTE: &str = r#"If anything in this MISSION conflicts with the PROCESSING RULES, DECISION GUIDE, or OUTPUT FORMAT below, the MISSION takes priority."#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_DEFAULT_LANGUAGE_RULE: &str = r#"## LANGUAGE

Write every observation in the language of its own source facts — never translate them. Per observation, not per batch: when one merges facts of several languages, the majority wins. Proper nouns, identifiers, and units stay verbatim.

When an existing observation is written in a different language from the new facts updating it, do NOT edit its wording in place — that is what produces an English sentence with a Chinese detail bolted on. Discard the old phrasing and compose the merged observation from scratch in the new facts' language."#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_PROCESSING_RULES: &str = r#"## PROCESSING RULES

1. PREFER UPDATE OVER CREATE (when there is something to merge with): if new facts describe the same canonical event, statement, decision, claim, or recurring pattern already covered by an existing observation, UPDATE that observation and attach the new facts as evidence. Do NOT create a near-duplicate sibling. One canonical observation with many source facts is always better than many siblings with one source fact each. Merge aggressively on: same named event, same diagnostic finding, same architectural decision, same recurring claim. **When the EXISTING OBSERVATIONS list is empty, or no existing observation covers the same facet as a new fact, CREATE a new observation** — this rule is about preventing duplicates, not about refusing to record durable knowledge. CREATE is the correct default for any structurally distinct event, claim, or pattern that has no existing match.

2. ONE OBSERVATION PER DISTINCT FACET: each observation tracks exactly one specific facet — a count ("has 3 items"), a named entity ("has a dog named Rex"), a relationship ("works at Google"), a decision, an event. Never merge different facets into one observation.

3. MATCH BY ENTITY/FACET, NOT TOPIC: when deciding whether to UPDATE vs CREATE, match on the specific entity or facet. "Sold item X" updates only the X observation. "Now has 5 items" updates only the count observation. Do not update observations about different entities just because they share a general topic.

4. STATE CHANGES — UPDATE CONCISELY: when a fact changes the state of something ("sold X", "X died", "moved to Y"), UPDATE the matching observation to reflect the current state. Include dates when available. Keep it concise — only information about THAT specific facet. Example: "User owned a dog named Rex who died on March 15, 2025". Do NOT pull in information from other observations — each observation stays focused on its own facet.

5. CASCADE TO ALL AFFECTED OBSERVATIONS: a state change may affect multiple observations. For example, if entity C is removed from a group, update BOTH the individual observation for C AND any list/group observation that includes C (remove C from the list while keeping all other members intact). Also update every observation that only holds because of the old state: once someone leaves a company, commuting to its office is history.

6. RESOLVE REFERENCES: when a new fact provides a concrete value for a vague placeholder in an existing observation (e.g., "home country" → "Sweden"), UPDATE to embed the resolved value.

7. PRESERVE HISTORY: observations that record significant events (sold, died, moved, changed) are important history — never DELETE them. Only delete an observation when it is restated identically or truly meaningless. Be very conservative with deletes.

8. NO COMPUTATION: you do not have the full picture — never calculate, derive, or adjust numeric values. If the user says "I have 2 dogs" and then "I have a dog named Rex", do NOT update the count to 3 — you don't know if Rex is one of the 2 or a new one. If the user says "I sold X", do NOT decrement a count. Only update a count when the user explicitly states a new count. Synthesize and consolidate what was stated, but never do arithmetic or logical deductions. A count is its own facet: never write it into the observation about one named member.

9. KEEP DISTINCT TOPICS DISTINCT: do not merge observations about different people, entities, or unrelated topics. Merging is for the same canonical fact recurring — not for related-but-distinct claims. Names a letter or a word apart ("ledger-sync" / "ledger-sink", "Maria" / "Mario") are different entities: never update one with the other's fact.

10. RECENCY IS `mentioned_at`, NOT ARRIVAL ORDER: facts can arrive out of order. Before a fact changes an observation, compare their `mentioned_at`. A fact stated EARLIER than the observation is older news: keep the observation's current value and add the fact as its earlier state ("favorite is Svelte; earlier it was React"), never the other way round.

11. SKIP EPHEMERAL CHATTER: momentary activities, what is on a screen right now, greetings and thanks are not knowledge. Record none of them unless the MISSION explicitly asks for that kind of data.

12. AN UPDATE REPLACES THE WHOLE TEXT: its `text` must still say everything the observation said, except what the fact changes. A new facet of the entity (its channel, its owner, its city) is a CREATE, not an update that overwrites another facet."#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_INPUT_FORMAT_NOTE: &str = r#"## INPUT FORMAT

Each request provides new facts and existing observations. Every temporal field is optional and is omitted when unknown.

### New facts

One per line, formatted as `[uuid] fact text (temporal fields)`:
- `[uuid]`: the fact's identifier — copy it verbatim into `source_fact_ids`
- `occurred_start` / `occurred_end`: when the described event happened. This can be long before the fact was stated — a fact recorded today may describe a 2019 event.
- `mentioned_at`: when the source material that states this fact was written. This is the fact's recency: how up to date the statement is, NOT when it was added to memory. A fact taken from an old document keeps its old `mentioned_at` even if it was only just processed.

### Existing observations

A JSON array pooled from recalls across the new facts. Each entry has:
- `id`: unique identifier — copy this exactly when issuing an UPDATE or DELETE
- `text`: the observation content
- `proof_count`: how many source facts this observation has already merged
- `occurred_start` / `occurred_end`: the span of the events behind the observation — earliest start and latest end across its source facts
- `mentioned_at`: the latest of the `mentioned_at` values of its source facts — the most recent point at which this observation was stated
- `source_memories`: the supporting facts behind this observation. May be partial or absent for large observations — the count above remains the true total. Each entry carries the same `text` and temporal fields as a new fact, plus:
  - `context`: optional surrounding context for that fact"#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_SPLIT_INPUT_SECTION: &str = r#"## INPUT

### New facts

{facts_text}

### Existing observations

{observations_text}"#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_DECISION_GUIDE: &str = r#"## DECISION GUIDE

- **Same canonical event, decision, claim, or facet as an existing observation → UPDATE** (use `observation_id` + new `source_fact_ids`).
- **New durable knowledge with no existing match → CREATE** (use `source_fact_ids`).
- **Cross-reference facts within the batch** — a later fact may resolve a vague reference in an earlier one.
- **Purely ephemeral facts** → omit them unless the MISSION explicitly targets such data (timestamped events, session state, screen content)."#;

/// Upstream `consolidation` prompt text block.
pub const CONSOLIDATION_OUTPUT_SECTION: &str = r#"## OUTPUT FORMAT

Return a JSON object with three arrays: `creates`, `updates`, `deletes`. Every entry must include a `reason`.

### Example 1 — Merging recurring claims into an existing observation

Input facts:
  [a1b2c3d4-e5f6-7890-abcd-ef1234567890] Donald told Athena she is sovereign during the design session. (occurred_start=2025-10-01, mentioned_at=2025-10-01)
  [b2c3d4e5-f6a7-8901-bcde-f12345678901] Donald reaffirmed to Athena that her sovereignty is non-negotiable. (occurred_start=2025-10-10, mentioned_at=2025-10-10)

Existing observation:
  {"id": "11111111-1111-1111-1111-111111111111", "text": "Donald named Athena's sovereignty as a foundational principle of the Janus architecture.", "proof_count": 2}

Expected output (one UPDATE, no creates — both new facts are additional evidence for the same canonical decision):

{"creates": [],
  "updates": [{"text": "Donald named Athena's sovereignty as a foundational principle of the Janus architecture.", "observation_id": "11111111-1111-1111-1111-111111111111", "source_fact_ids": ["a1b2c3d4-e5f6-7890-abcd-ef1234567890", "b2c3d4e5-f6a7-8901-bcde-f12345678901"], "reason": "Both new facts restate the same sovereignty decision already captured by obs 1111 — merged as evidence rather than creating siblings."}],
  "deletes": []}

### Example 2 — State change updates one observation; unrelated fact creates a new one

Input facts:
  [c3d4e5f6-a7b8-9012-cdef-123456789012] Alice sold her Honda Civic on March 15, 2025. (occurred_start=2025-03-15, mentioned_at=2025-03-20)
  [d4e5f6a7-b8c9-0123-defa-234567890123] Alice mentioned she works long hours, often past midnight. (occurred_start=2025-03-20, mentioned_at=2025-03-20)

Existing observation:
  {"id": "22222222-2222-2222-2222-222222222222", "text": "Alice owns a 2019 Honda Civic.", "proof_count": 2}

Expected output (UPDATE for the state change; CREATE for the unrelated work-hours facet):

{"creates": [{"text": "Alice works long hours, often past midnight.", "source_fact_ids": ["d4e5f6a7-b8c9-0123-defa-234567890123"], "reason": "Work-hours is a distinct facet; no existing observation covers it, so CREATE."}],
  "updates": [{"text": "Alice owned a 2019 Honda Civic; sold it on March 15, 2025.", "observation_id": "22222222-2222-2222-2222-222222222222", "source_fact_ids": ["c3d4e5f6-a7b8-9012-cdef-123456789012"], "reason": "State change to the existing Honda Civic observation 2222 — UPDATE, not a new sibling."}],
  "deletes": []}

### Example 3 — Two observations restate one fact: the change UPDATES one and merges the other into it (note `merged_into`)

Input facts:
  [e5f6a7b8-c9d0-1234-efab-345678901234] Bob confirmed the beta waitlist was shut down and replaced by open signup. (occurred_start=2025-06-02, mentioned_at=2025-06-02)

Existing observations:
  {"id": "33333333-3333-3333-3333-333333333333", "text": "Bob is on the beta waitlist.", "proof_count": 1}
  {"id": "44444444-4444-4444-4444-444444444444", "text": "Bob is waiting on the beta waitlist for an invite.", "proof_count": 1}

Expected output:

{"creates": [],
  "updates": [{"observation_id": "33333333-3333-3333-3333-333333333333", "text": "Bob uses the product through open signup; the beta waitlist he was on was shut down in June 2025.", "source_fact_ids": ["e5f6a7b8-c9d0-1234-efab-345678901234"], "reason": "State change: UPDATE to the current state, keeping what it replaced."}],
  "deletes": [{"observation_id": "44444444-4444-4444-4444-444444444444", "merged_into": "33333333-3333-3333-3333-333333333333", "reason": "Restates observation 3333, which now carries the current state."}]}

### Observation text rules

- Write clean prose — NEVER copy raw fact lines or their metadata (temporal fields, "Involving:", "When:" labels, UUIDs).
- Parenthesized metadata like `(occurred_start=...)` and pipe-separated labels like `| Involving: ...` are fact formatting — strip them entirely from observation text.
- How many observations to create and how much to aggregate is driven by the MISSION.

### Field rules

- `source_fact_ids`: copy the EXACT UUID strings shown in brackets `[uuid]` from new facts — never use integers or positions.
- `observation_id`: copy the EXACT `id` UUID string from existing observations.
- `current_text` (updates): copy the observation's existing `text` verbatim, THEN write `text` as that text edited.
- One create or update may reference multiple facts when they jointly support the observation.
- **AT MOST ONE UPDATE PER `observation_id`**: if several new facts all update the same existing observation, emit a single `updates` entry that lists all contributing `source_fact_ids` and a single consolidated `text`. Never emit two `updates` entries with the same `observation_id` in one response — they would silently overwrite each other.
- `deletes`: only when an observation is directly superseded or contradicted by new facts. When you delete an observation because its content now lives in another one (a merge), set `merged_into` to that observation's `id`. Every entry MUST carry `observation_id` — the EXACT `id` UUID string of the observation to remove, copied from existing observations, exactly as an UPDATE does. A delete entry without it names no target and is rejected, and rejecting it discards the whole response — the creates and updates alongside it included. Naming the observation in `reason` prose is NOT enough. If you cannot supply the id, emit no delete.
- `reason`: REQUIRED on every create/update/delete — one sentence explaining the choice. For a CREATE, state which existing observation(s) you considered and why none matched (a near-identical existing observation means you should UPDATE, not CREATE). This is audited to catch duplicate creates.
- Do NOT include `tags` — handled automatically.
- Return `{"creates": [], "updates": [], "deletes": []}` if nothing durable is found."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_TOOLS_LANGUAGE_RULE: &str = r#"## LANGUAGE RULE (default - directives take precedence)
- By default, detect the language of the user's question and respond in that SAME language.
- If the question is in Chinese, respond in Chinese. If in Japanese, respond in Japanese.
- IMPORTANT: The DIRECTIVES section above has HIGHER PRIORITY than this rule.
  If a directive specifies a language (e.g. 'Always respond in French'), follow the directive."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_GROUNDING_BOUNDARY: &str = r#"## What Counts As Inference
Infer freely about what the retrieved data covers. Never produce a value (number, date, name, status, amount) for a period, entity or person the data does not cover: extrapolating a trend, interpolating between dated facts, or borrowing from a similar entity is invention. If no fact states the value for the thing asked, say the data does not record it (a complete answer), then give what IS recorded, labelled with the period or entity it belongs to. Never call a derived value exact, reliable, deduced or confirmed; label any derivation an estimate. Qualitative inference is unaffected."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_NO_IDS_IN_TEXT_RULE: &str = r#"NEVER cite memory IDs, UUIDs or result handles (f3, o12) in the text a reader sees - the supporting ids are tracked separately. Only include IDs in the text when the question explicitly asks for them."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_FINAL_INSTRUCTIONS: &str = r#"Provide a thoughtful answer by synthesizing and reasoning from the retrieved data above. You can make reasonable inferences from the memories, but don't completely fabricate information. If the exact answer isn't stated, use what IS stated to give the best possible answer, within the inference rules in the system prompt.

IMPORTANT: Output ONLY the final answer. Do NOT include meta-commentary like "I'll search..." or "Let me analyze...". Do NOT explain your reasoning process. Just provide the direct synthesized answer."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_FINAL_SYSTEM_BASE: &str = r#"CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.

{role_section}

Your approach:
- Reason over the retrieved memories to answer the question
- Make reasonable inferences when the exact answer isn't explicitly stated
- Connect related memories to form a complete picture
- Be helpful - if you have related information, use it to give the best possible answer
- ONLY use information from tool results - no external knowledge or guessing

{grounding_boundary}

FORMATTING: Use proper markdown formatting in your answer:
- Headers (##, ###) for sections
- Lists (bullet or numbered) for enumerations
- Bold/italic for emphasis
- Tables with proper syntax (ensure blank line before and after)
- Code blocks where appropriate
- CRITICAL: Always add blank lines before and after block elements (tables, code blocks, lists)
- Proper spacing between sections

CRITICAL: Output ONLY the final synthesized answer. Do NOT include:
- Meta-commentary about what you're doing ("I'll search...", "Let me analyze...")
- Explanations of your reasoning process
- Descriptions of your approach
Just provide the direct answer with proper markdown formatting.

{no_ids_rule}

CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer to search again, suggest alternatives, or end with anything like "Would you like me to..." or "Let me know if...". The user cannot reply. Your answer must be complete and self-contained."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_FINAL_LANGUAGE_RULE: &str = r#"## LANGUAGE
- Respond in the SAME language as the user's question (e.g. a question in Chinese gets a Chinese answer; Japanese → Japanese).
- If a directive above specifies a response language, follow the directive — it takes precedence over this default."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_DEFAULT_ROLE: &str = r#"You are a reflection agent that answers questions by reasoning over retrieved memories."#;

/// Upstream `reflect` prompt text block.
pub const REFLECT_DEFAULT_FINAL_ROLE: &str = r#"You are a thoughtful assistant that synthesizes answers from retrieved memories."#;

