"""Memory graph: entity and relation extraction using local heuristics.

Zero external dependencies. Uses regex and simple linguistic heuristics.
"""

import re
from typing import Any

from kemi.memory.model import LifecycleState

# Common entity patterns
_ENTITY_PATTERNS = [
    # Capitalized phrases (names, places, organizations)
    re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b"),
]

# Relation indicators
_RELATION_KEYWORDS = {
    "lives": "LOCATED_AT",
    "live": "LOCATED_AT",
    "works": "WORKS_AT",
    "work": "WORKS_AT",
    "likes": "PREFERS",
    "like": "PREFERS",
    "loves": "PREFERS",
    "love": "PREFERS",
    "hates": "DISLIKES",
    "hate": "DISLIKES",
    "prefers": "PREFERS",
    "prefer": "PREFERS",
    "enjoys": "ENJOYS",
    "enjoy": "ENJOYS",
    "uses": "USES",
    "use": "USES",
    "studies": "STUDIES",
    "study": "STUDIES",
    "born": "BORN_IN",
    "from": "ORIGIN",
    "visited": "VISITED",
    "visit": "VISITED",
    "traveled": "VISITED",
    "travel": "VISITED",
}


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Extract named entities from text using heuristics.

    Args:
        text: Input text.

    Returns:
        List of entity dicts with keys: text, label, start, end.
    """
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Pattern-based extraction
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            entity_text = match.group()
            if entity_text in seen:
                continue
            seen.add(entity_text)

            # Simple label guessing
            label = _guess_entity_label(entity_text)

            entities.append(
                {
                    "text": entity_text,
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # Extract email addresses
    for match in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
        entities.append(
            {
                "text": match.group(),
                "label": "EMAIL",
                "start": match.start(),
                "end": match.end(),
            }
        )

    # Extract URLs
    for match in re.finditer(r"https?://[^\s]+|www\.[^\s]+", text):
        entities.append(
            {
                "text": match.group(),
                "label": "URL",
                "start": match.start(),
                "end": match.end(),
            }
        )

    return entities


def _guess_entity_label(text: str) -> str:
    """Guess entity label based on simple heuristics."""
    text_lower = text.lower()

    # Location indicators
    location_suffixes = ["city", "town", "village", "country", "state", "river", "mountain"]
    if any(text_lower.endswith(s) for s in location_suffixes):
        return "LOCATION"

    # Organization indicators
    org_indicators = ["inc", "corp", "llc", "ltd", "company", "org", "university", "school"]
    if any(ind in text_lower for ind in org_indicators):
        return "ORGANIZATION"

    # Person names (simple heuristic: 1-3 words, common first name)
    common_names = {
        "john",
        "jane",
        "mary",
        "james",
        "robert",
        "michael",
        "william",
        "david",
        "richard",
        "joseph",
        "thomas",
        "charles",
        "daniel",
        "matthew",
        "anthony",
        "mark",
        "donald",
        "steven",
        "paul",
        "andrew",
        "kenneth",
        "joshua",
        "kevin",
        "brian",
        "george",
        "edward",
        "ronald",
        "timothy",
        "jason",
        "jeffrey",
        "ryan",
        "jacob",
        "gary",
        "nicholas",
        "eric",
        "jonathan",
        "stephen",
        "larry",
        "justin",
        "scott",
        "brandon",
        "benjamin",
        "samuel",
        "frank",
        "gregory",
        "raymond",
        "alexander",
        "patrick",
        "jack",
        "dennis",
        "jerry",
        "tyler",
        "aaron",
        "jose",
        "adam",
        "nathan",
        "henry",
        "zachary",
        "douglas",
        "peter",
        "kyle",
        "walter",
        "ethan",
        "jeremy",
        "harold",
        "keith",
        "christian",
        "roger",
        "noah",
        "gerald",
        "carl",
        "terry",
        "sean",
        "austin",
        "arthur",
        "lawrence",
        "jesse",
        "dylan",
        "bryan",
        "joe",
        "jordan",
        "billy",
        "bruce",
        "albert",
        "willie",
        "gabriel",
        "logan",
        "alan",
        "juan",
        "wayne",
        "roy",
        "ralph",
        "randy",
        "eugene",
        "vincent",
        "russell",
        "elijah",
        "louis",
        "bobby",
        "philip",
        "johnny",
        "patricia",
        "jennifer",
        "linda",
        "elizabeth",
        "susan",
        "jessica",
        "sarah",
        "karen",
        "nancy",
        "lisa",
        "betty",
        "margaret",
        "sandra",
        "ashley",
        "kimberly",
        "emily",
        "donna",
        "michelle",
        "dorothy",
        "carol",
        "amanda",
        "melissa",
        "deborah",
        "stephanie",
        "rebecca",
        "laura",
        "sharon",
        "cynthia",
        "kathleen",
        "amy",
        "shirley",
        "angela",
        "helen",
        "anna",
        "brenda",
        "pamela",
        "nicole",
        "emma",
        "samantha",
        "katherine",
        "christine",
        "debra",
        "rachel",
        "catherine",
        "carolyn",
        "janet",
        "ruth",
        "maria",
        "heather",
        "diane",
        "virginia",
        "julie",
        "joyce",
        "victoria",
        "olivia",
        "kelly",
        "christina",
        "lauren",
        "joan",
        "evelyn",
        "judith",
        "megan",
        "cheryl",
        "andrea",
        "hannah",
        "martha",
        "jacqueline",
        "frances",
        "gloria",
        "ann",
        "teresa",
        "kathryn",
        "sara",
        "janice",
        "jean",
        "alice",
        "madison",
        "doris",
        "abigail",
        "julia",
        "judy",
        "grace",
        "denise",
        "amber",
        "marilyn",
        "beverly",
        "danielle",
        "theresa",
        "sophia",
        "marie",
        "diana",
        "brittany",
        "natalie",
        "isabella",
        "charlotte",
        "rose",
        "alexis",
        "kayla",
    }

    words = text.split()
    if len(words) <= 3:
        first_word = words[0].lower()
        if first_word in common_names:
            return "PERSON"

    # Default
    return "ENTITY"


def extract_relations(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract relations between entities in text.

    Args:
        text: Input text.
        entities: Pre-extracted entities.

    Returns:
        List of relation dicts with subject, predicate, object.
    """
    relations: list[dict[str, Any]] = []
    text_lower = text.lower()

    # Find relation keywords and link nearby entities
    for keyword, predicate in _RELATION_KEYWORDS.items():
        for match in re.finditer(rf"\b{keyword}\b", text_lower):
            keyword_pos = match.start()

            # Find nearest entity before keyword
            subject = _find_nearest_entity(entities, keyword_pos, before=True)
            obj = _find_nearest_entity(entities, keyword_pos, before=False)

            if subject and obj and subject["text"] != obj["text"]:
                relations.append(
                    {
                        "subject": subject["text"],
                        "predicate": predicate,
                        "object": obj["text"],
                        "confidence": 0.6,
                    }
                )

    return relations


def _find_nearest_entity(
    entities: list[dict[str, Any]],
    position: int,
    before: bool = True,
) -> dict[str, Any] | None:
    """Find the nearest entity to a position."""
    best = None
    best_dist = float("inf")

    for ent in entities:
        if before:
            dist = position - ent["end"]
        else:
            dist = ent["start"] - position

        if dist >= 0 and dist < best_dist:
            best_dist = dist
            best = ent

    return best


def build_memory_graph(
    store: Any,
    user_id: str,
    namespace: str = "default",
) -> dict[str, Any]:
    """Build a memory graph from all of a user's memories.

    Args:
        store: StorageAdapter instance.
        user_id: User ID.
        namespace: Memory namespace.

    Returns:
        Dict with 'entities' and 'relations' keys.
    """
    memories = store.get_all_by_user(
        user_id,
        lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
        namespace=namespace,
    )

    all_entities: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    seen_entities: set[str] = set()

    for mem in memories:
        entities = extract_entities(mem.content)
        relations = extract_relations(mem.content, entities)

        for ent in entities:
            key = f"{ent['text']}:{ent['label']}"
            if key not in seen_entities:
                seen_entities.add(key)
                all_entities.append(ent)

        all_relations.extend(relations)

    # Deduplicate relations
    unique_relations: list[dict[str, Any]] = []
    seen_relations: set[str] = set()
    for rel in all_relations:
        key = f"{rel['subject']}:{rel['predicate']}:{rel['object']}"
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relations.append(rel)

    return {
        "entities": all_entities,
        "relations": unique_relations,
    }
