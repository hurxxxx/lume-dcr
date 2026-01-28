#!/usr/bin/env python3
import argparse
from pathlib import Path
import difflib


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize(text: str) -> str:
    # collapse whitespace for fair comparison
    return "".join(text.split())


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True, help="reference text file (pdf text layer)")
    parser.add_argument("--candidate", type=str, required=True, action="append", help="candidate text/markdown file")
    args = parser.parse_args()

    ref_text = normalize(read_text(Path(args.ref)))
    if not ref_text:
        raise SystemExit("Reference text is empty.")

    print(f"Reference length (normalized): {len(ref_text)}")

    for cand_path in args.candidate:
        path = Path(cand_path)
        cand_text = normalize(read_text(path))
        score = similarity(ref_text, cand_text)
        print(f"{path.name}: len={len(cand_text)} similarity={score:.4f}")


if __name__ == "__main__":
    main()
