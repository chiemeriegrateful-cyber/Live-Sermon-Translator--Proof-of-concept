# scripts/route.py
import re

BOOKS_DE = {
    "genesis", "exodus", "levitikus", "numeri", "deuteronomium",
    "josua", "richter", "ruth", "samuel", "könige", "chronik", "esra",
    "nehemia", "esther", "hiob", "psalm", "psalmen", "sprüche", "prediger",
    "hohelied", "jesaja", "jeremia", "klagelieder", "hesekiel", "ezechiel",
    "daniel", "hosea", "joel", "amos", "obadja", "jona", "micha", "nahum",
    "habakuk", "zephanja", "haggai", "sacharja", "maleachi",
    "matthäus", "markus", "lukas", "johannes", "apostelgeschichte",
    "römer", "korinther", "galater", "epheser", "philipper", "kolosser",
    "thessalonicher", "timotheus", "titus", "philemon", "hebräer",
    "jakobus", "petrus", "judas", "offenbarung",
}

VERSE_PATTERN = re.compile(r"\b\d+\s*[,:]\s*\d+")
BOOK_PATTERN = re.compile(r"\b(" + "|".join(BOOKS_DE) + r")\b", re.IGNORECASE)

WORD_COUNT_THRESHOLD = 40


def route(segment: str, previous_decision: str = None) -> str:
    """Return 'llm' or 'opus' for this segment."""
    if BOOK_PATTERN.search(segment):
        return "llm"
    if VERSE_PATTERN.search(segment):
        return "llm"
    if len(segment.split()) > WORD_COUNT_THRESHOLD:
        return "llm"
    # Scripture announcement in the previous segment usually leads
    # into the actual verse being quoted here.
    if previous_decision == "llm-scripture":
        return "llm"
    return "opus"


def classify(segment: str) -> str:
    """Return the raw category for use in stateful routing."""
    if BOOK_PATTERN.search(segment) or VERSE_PATTERN.search(segment):
        return "llm-scripture"
    if len(segment.split()) > WORD_COUNT_THRESHOLD:
        return "llm-long"
    return "opus"