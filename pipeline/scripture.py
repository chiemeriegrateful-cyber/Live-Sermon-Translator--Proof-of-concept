# pipeline/scripture.py
"""Fuzzy-match a German segment against stored Bible verses.

The result is shown next to the live translation as verified text, so a reader
can see when the translator's rendering of a quoted verse drifts from the
canonical one. Data comes from scripts/build_scripture.py.
"""
import json
import re
from bisect import bisect_left
from pathlib import Path

from rapidfuzz import fuzz, process

SCRIPTURE_PATH = Path("data/scripture/scripture.json")

# A fragment shorter than this is too generic to identify a verse.
MIN_CHARS = 25
# Whole-segment vs whole-verse similarity (segment is about one verse long).
# Two tiers, so the display never claims more than the score supports.
# Provisional: the only real calibration data so far is data/transcripts/
# sermon_reference.txt, where two real quotes in modern Luther wording score
# 74-75 against the 1912 text and the best non-quote scores 59. Two positives
# is not enough to trust these numbers; revisit with more labelled sermons.
RATIO_POSSIBLE = 65   # shown greyed as "possible match"
RATIO_CONFIDENT = 80  # shown as verified
# Segment is a fragment of a longer verse; a substring match must be tighter.
PARTIAL_CONFIDENT = 90
# A "possible" match on a short segment is noise ("Die Menschen kamen und
# gingen." scored 66 against Ecclesiastes 3:4). The real quotes seen so far are
# 13+ words, the false positive 5, so short segments get no possible-tier box.
POSSIBLE_MIN_WORDS = 8
# A segment that fits several verses about equally well identifies none of
# them (formulaic phrases like "Und der HERR redete mit Mose und sprach"), so
# no box is shown. A fragment that is a verbatim substring of exactly one verse
# ("Und Jesus ging in den Tempel." -> Mark 11:15) is still shown as verified.
AMBIGUITY_MARGIN = 2

BOOK_NAMES = {
    "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus", "NUM": "Numbers",
    "DEU": "Deuteronomy", "JOS": "Joshua", "JDG": "Judges", "RUT": "Ruth",
    "1SA": "1 Samuel", "2SA": "2 Samuel", "1KI": "1 Kings", "2KI": "2 Kings",
    "1CH": "1 Chronicles", "2CH": "2 Chronicles", "EZR": "Ezra",
    "NEH": "Nehemiah", "EST": "Esther", "JOB": "Job", "PSA": "Psalm",
    "PRO": "Proverbs", "ECC": "Ecclesiastes", "SOL": "Song of Solomon",
    "ISA": "Isaiah", "JER": "Jeremiah", "LAM": "Lamentations",
    "EZE": "Ezekiel", "DAN": "Daniel", "HOS": "Hosea", "JOE": "Joel",
    "AMO": "Amos", "OBA": "Obadiah", "JON": "Jonah", "MIC": "Micah",
    "NAH": "Nahum", "HAB": "Habakkuk", "ZEP": "Zephaniah", "HAG": "Haggai",
    "ZEC": "Zechariah", "MAL": "Malachi", "MAT": "Matthew", "MAR": "Mark",
    "LUK": "Luke", "JOH": "John", "ACT": "Acts", "ROM": "Romans",
    "1CO": "1 Corinthians", "2CO": "2 Corinthians", "GAL": "Galatians",
    "EPH": "Ephesians", "PHI": "Philippians", "COL": "Colossians",
    "1TH": "1 Thessalonians", "2TH": "2 Thessalonians", "1TI": "1 Timothy",
    "2TI": "2 Timothy", "TIT": "Titus", "PHM": "Philemon", "HEB": "Hebrews",
    "JAM": "James", "1PE": "1 Peter", "2PE": "2 Peter", "1JO": "1 John",
    "2JO": "2 John", "3JO": "3 John", "JUD": "Jude", "REV": "Revelation",
}

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, and fold Luther-1912 spellings so they
    compare against modern ASR output (daß/dass, ß/ss)."""
    text = text.lower().replace("ß", "ss")
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def display_ref(ref: str) -> str:
    code, chapter_verse = ref.split(" ", 1)
    return f"{BOOK_NAMES.get(code, code)} {chapter_verse}"


class ScriptureIndex:
    def __init__(self, path: Path = SCRIPTURE_PATH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = sorted(
            ((normalize(v["de"]), ref) for ref, v in raw.items()),
            key=lambda r: len(r[0]),
        )
        self._raw = raw
        self._texts = [r[0] for r in rows]
        self._refs = [r[1] for r in rows]
        self._lengths = [len(t) for t in self._texts]

    def find(self, segment: str) -> dict | None:
        """Best verse for this segment, or None if nothing is close enough."""
        query = normalize(segment)
        if len(query) < MIN_CHARS:
            return None

# (score, verse index, scorer). Two per scorer so a near-tie is visible.
        scored = [
            (score, idx, "ratio")
            for _, score, idx in process.extract(
                query, self._texts, scorer=fuzz.ratio,
                score_cutoff=RATIO_POSSIBLE, limit=2,
            )
        ]
        # Fragment of a longer verse: only compare against verses at least as
        # long as the segment, otherwise any short verse "matches".
        start = bisect_left(self._lengths, int(len(query) * 0.9))
        scored += [
            (score, idx + start, "partial")
            for _, score, idx in process.extract(
                query, self._texts[start:], scorer=fuzz.partial_ratio,
                score_cutoff=PARTIAL_CONFIDENT, limit=2,
            )
        ]
        if not scored:
            return None

        scored.sort(key=lambda s: s[0], reverse=True)
        best_score, best_idx, best_scorer = scored[0]
        # Another verse scoring about as well means the segment identifies none
        # of them (identical verses such as "Und der HERR redete mit Mose und
        # sprach:" recur dozens of times).
        ambiguous = any(
            other_idx != best_idx and other_score >= best_score - AMBIGUITY_MARGIN
            for other_score, other_idx, _ in scored[1:]
        )
        if ambiguous:
            # Picking one of several equally good verses would be a guess
            # presented as a citation, so show nothing.
            return None
        confident = best_scorer == "partial" or best_score >= RATIO_CONFIDENT
        if not confident and len(segment.split()) < POSSIBLE_MIN_WORDS:
            return None
        ref = self._refs[best_idx]
        verse = self._raw[ref]
        return {
            "ref": display_ref(ref),
            "de": verse["de"],
            "en": verse["en"],
            "score": round(best_score),
            "tier": "confident" if confident else "possible",
            "source": "Luther 1912 / WEB",
        }


def load_index() -> ScriptureIndex | None:
    """None if the data file has not been built; lookups are then skipped."""
    if not SCRIPTURE_PATH.exists():
        print(f"scripture: {SCRIPTURE_PATH} missing, run scripts/build_scripture.py")
        return None
    return ScriptureIndex()
