"""CLI: ingest local knowledge base into chunked JSON.

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --kb-dir knowledge_base --out data/rag_chunks.json
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `app.*` importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag import ingest_knowledge_base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local knowledge base.")
    parser.add_argument("--kb-dir", default="knowledge_base")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = ingest_knowledge_base(args.kb_dir, args.out)
    print(
        "Ingested {n} chunks from {f} files ({pdf} PDF, {txt} text) -> {p}".format(
            n=result.chunk_count,
            f=result.files_processed,
            pdf=result.pdf_files,
            txt=result.text_files,
            p=result.output_path,
        )
    )
    if result.skipped_files:
        print("Skipped {n} file(s) with no extractable text.".format(n=result.skipped_files))
    if result.chunk_count == 0:
        print(
            "Knowledge base is empty or unreadable. The app will still run; "
            "drop .md/.txt/.pdf files into knowledge_base/ and re-run this script."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
