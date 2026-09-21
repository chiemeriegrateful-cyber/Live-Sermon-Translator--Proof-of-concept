# scripts/route.py
import re

BOOKS_DE = {
    # Torah — every announcement form Luther-tradition churches use.
    "genesis", "exodus", "levitikus", "numeri", "deuteronomium",
    "1. mose", "2. mose", "3. mose", "4. mose", "5. mose",
    "1. buch mose", "2. buch mose", "3. buch mose", "4. buch mose", "5. buch mose",
    "erstes buch mose", "zweites buch mose", "drittes buch mose",
    "viertes buch mose", "fünftes buch mose",
    "buch mose",  # fallback for ASR dropping the number

    # Historical books.
    "josua", "buch josua",
    "richter", "buch der richter",
    "ruth", "buch ruth",
    "1. samuel", "2. samuel", "samuel",
    "erstes buch samuel", "zweites buch samuel",
    "1. könige", "2. könige", "könige",
    "erstes buch der könige", "zweites buch der könige",
    "1. chronik", "2. chronik", "chronik",
    "erstes buch der chronik", "zweites buch der chronik",
    "esra", "buch esra",
    "nehemia", "buch nehemia",
    "esther", "ester", "buch ester",

    # Wisdom and poetry.
    "hiob", "buch hiob",
    "psalm", "psalmen", "buch der psalmen",
    "sprüche", "sprichwörter", "buch der sprüche",
    "prediger", "kohelet",
    "hohelied", "hoheslied", "hohes lied", "das hohelied salomos",

    # Major prophets.
    "jesaja", "buch jesaja", "der prophet jesaja",
    "jeremia", "buch jeremia", "der prophet jeremia",
    "klagelieder", "klagelieder jeremias",
    "hesekiel", "ezechiel", "buch hesekiel", "buch ezechiel",
    "daniel", "buch daniel",

    # Minor prophets.
    "hosea", "joel", "amos", "obadja", "jona", "micha", "nahum",
    "habakuk", "zephanja", "haggai", "sacharja", "maleachi",
    "der prophet hosea", "der prophet joel", "der prophet amos",
    "der prophet obadja", "der prophet jona", "der prophet micha",
    "der prophet nahum", "der prophet habakuk", "der prophet zephanja",
    "der prophet haggai", "der prophet sacharja", "der prophet maleachi",

    # Gospels and Acts.
    "matthäus", "markus", "lukas", "johannes",
    "evangelium nach matthäus", "evangelium nach markus",
    "evangelium nach lukas", "evangelium nach johannes",
    "matthäusevangelium", "markusevangelium",
    "lukasevangelium", "johannesevangelium",
    "apostelgeschichte", "apg",

    # Pauline epistles.
    "römer", "brief an die römer", "römerbrief",
    "1. korinther", "2. korinther", "korinther",
    "erster brief an die korinther", "zweiter brief an die korinther",
    "erster korintherbrief", "zweiter korintherbrief",
    "galater", "brief an die galater", "galaterbrief",
    "epheser", "brief an die epheser", "epheserbrief",
    "philipper", "brief an die philipper", "philipperbrief",
    "kolosser", "brief an die kolosser", "kolosserbrief",
    "1. thessalonicher", "2. thessalonicher", "thessalonicher",
    "erster brief an die thessalonicher", "zweiter brief an die thessalonicher",
    "1. timotheus", "2. timotheus", "timotheus",
    "erster brief an timotheus", "zweiter brief an timotheus",
    "titus", "brief an titus",
    "philemon", "brief an philemon",

    # General epistles.
    "hebräer", "brief an die hebräer", "hebräerbrief",
    "jakobus", "jakobusbrief",
    "1. petrus", "2. petrus", "petrus",
    "erster petrusbrief", "zweiter petrusbrief",
    "1. johannes", "2. johannes", "3. johannes",
    "erster johannesbrief", "zweiter johannesbrief", "dritter johannesbrief",
    "judas", "judasbrief",

    # Apocalypse.
    "offenbarung", "offenbarung des johannes",
}

# Longest-first so multi-word phrases match before their subphrases
# ("Buch Mose" before "Mose", "Erstes Buch Mose" before "Buch Mose").
_BOOKS_SORTED = sorted(BOOKS_DE, key=len, reverse=True)
BOOK_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(b) for b in _BOOKS_SORTED) + r")(?!\w)",
    re.IGNORECASE,
)

# Catches numeric "3,16" or "3:16" plus the spoken forms preachers actually use.
VERSE_PATTERN = re.compile(
    r"\b\d+\s*[,:]\s*\d+"                             # 3,16 / 3:16
    r"|\bkapitel\s+\d+"                                # Kapitel 3
    r"|\bverse?\s+\d+(?:\s*(?:bis|-|–)\s*\d+)?"        # Vers 5 / Verse 1 bis 19
    r"|\bab\s+vers\s+\d+",                             # ab Vers 5
    re.IGNORECASE,
)

WORD_COUNT_THRESHOLD = 40
SCRIPTURE_TTL = 15  # segments to keep routing to llm after a scripture cue

# Preachers deliver punchy fragments ("Da war viel Volk. Ein großes Getümmel!").
# Alone, each is meaningless to opus (no context); gemma sees the previous
# segments and can repair ASR misses. So a run of short segments goes to llm.
SHORT_SEGMENT_WORDS = 6
SHORT_RUN_TO_LLM = 2  # this many consecutive short segments (incl. current)


def classify(segment: str) -> str:
    """Raw category for this segment in isolation."""
    if BOOK_PATTERN.search(segment) or VERSE_PATTERN.search(segment):
        return "llm-scripture"
    if len(segment.split()) > WORD_COUNT_THRESHOLD:
        return "llm-long"
    return "opus"


def route(segment: str, state: dict | None = None) -> tuple[str, dict]:
    """Decide 'llm' or 'opus' for this segment and return updated state.

    state is a dict carrying rolling context between calls. Pass {} the
    first time; pass back whatever this function returns on subsequent calls.
    """
    if state is None:
        state = {}
    new_state = dict(state)

    cls = classify(segment)

    if cls == "llm-scripture":
        new_state["scripture_ttl"] = SCRIPTURE_TTL
        return "llm", new_state

    ttl = new_state.get("scripture_ttl", 0)
    if ttl > 0:
        new_state["scripture_ttl"] = ttl - 1
        return "llm", new_state

    if cls == "llm-long":
        new_state["short_run"] = 0
        return "llm", new_state

    is_short = len(segment.split()) <= SHORT_SEGMENT_WORDS
    short_run = new_state.get("short_run", 0) + 1 if is_short else 0
    new_state["short_run"] = short_run
    if is_short and short_run >= SHORT_RUN_TO_LLM:
        return "llm", new_state

    # German nouns are capitalised, so a segment opening in lowercase ("um ein
    # Opfer zu bringen") is the tail of a clipped sentence and needs context.
    if segment[:1].islower():
        return "llm", new_state

    return "opus", new_state