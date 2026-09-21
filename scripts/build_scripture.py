"""Build data/scripture/scripture.json from public-domain verse-per-line texts.

German: Luther 1912 (deu1912). English: World English Bible (eng-web).
Both come from ebible.org and are public domain.

    python scripts/build_scripture.py
"""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "de": ("https://ebible.org/Scriptures/deu1912_vpl.zip", "deu1912_vpl.txt"),
    "en": ("https://ebible.org/Scriptures/eng-web_vpl.zip", "eng-web_vpl.txt"),
}
OUT_PATH = Path("data/scripture/scripture.json")


def fetch_verses(url: str, member: str) -> dict[str, str]:
    # ebible.org returns 403 to urllib's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        text = zf.read(member).decode("utf-8-sig")
    verses = {}
    for line in text.splitlines():
        # "MRK 11:1 Und da sie ..." -> ref "MRK 11:1"
        parts = line.split(" ", 2)
        if len(parts) == 3 and ":" in parts[1]:
            verses[f"{parts[0]} {parts[1]}"] = parts[2].strip()
    return verses


def main():
    de = fetch_verses(*SOURCES["de"])
    en = fetch_verses(*SOURCES["en"])
    # The German side drives matching; English is attached where the
    # versification lines up and left out where it does not.
    out = {ref: {"de": t, "en": en.get(ref)} for ref, t in de.items()}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    missing = sum(1 for v in out.values() if v["en"] is None)
    print(f"wrote {len(out)} verses to {OUT_PATH} ({missing} without English)")


if __name__ == "__main__":
    main()
