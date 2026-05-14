"""Chunk + embed the cleaned UiA corpus and persist a FAISS index.

    python scripts/04_build_uia_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RagPipeline, load_config


def load_uia_pages(jsonl_path: Path) -> list[dict]:
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append({
                "id": rec["id"],
                "text": rec["text"],
                "url": rec.get("url"),
                "title": rec.get("title"),
            })
    return docs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    pages_path = repo_root / config["paths"]["uia_clean_dir"] / "pages.jsonl"
    if not pages_path.exists():
        print(f"No cleaned pages at {pages_path}. Run 03_scrape_uia.py first.")
        return

    docs = load_uia_pages(pages_path)
    pipeline = RagPipeline(config)
    chunks = pipeline.build_index_from_documents(docs)

    out_dir = repo_root / config["paths"]["index_dir"] / "uia"
    pipeline.save_index(out_dir)
    print(f"Saved {len(chunks)} chunks from {len(docs)} pages to {out_dir}.")


if __name__ == "__main__":
    main()
