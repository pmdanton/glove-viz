#!/usr/bin/env python3
"""Extract the curated word subset from the full GloVe 300d file.

Run this once if you need to regenerate data/glove_subset.txt.
Requires the full glove.6B.300d.txt in data/ (not committed).

Usage:
    python extract_subset.py
"""

from pathlib import Path

from src.glove_viz.wordlist import WORDS

GLOVE_FULL = Path("data/glove.6B.300d.txt")
SUBSET_OUT = Path("data/glove_subset.txt")


def main() -> None:
    if not GLOVE_FULL.exists():
        print(f"Full GloVe file not found: {GLOVE_FULL}")
        print("Download from: https://nlp.stanford.edu/data/glove.6B.zip")
        return

    found: dict[str, str] = {}
    with open(GLOVE_FULL) as f:
        for line in f:
            word = line.strip().split()[0]
            if word in WORDS:
                found[word] = line.strip()

    with open(SUBSET_OUT, "w") as out:
        for word in sorted(found):
            out.write(found[word] + "\n")

    missing = WORDS - set(found.keys())
    print(f"Extracted {len(found)}/{len(WORDS)} words → {SUBSET_OUT}")
    if missing:
        print(f"Missing: {sorted(missing)}")


if __name__ == "__main__":
    main()
