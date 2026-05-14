"""Interactive CLI chatbot over the UiA index.

    python scripts/05_run_chatbot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RagPipeline, load_config


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")

    index_dir = repo_root / config["paths"]["index_dir"] / "uia"
    if not index_dir.exists():
        print(f"No UiA index at {index_dir}. Run 04_build_uia_index.py first.")
        return

    pipeline = RagPipeline(config)
    pipeline.load_index(index_dir)
    print("UiA chatbot ready. Type a question, or 'quit' to exit.\n")

    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            break
        ans = pipeline.answer(q)
        print(f"\nbot> {ans.answer}\n")
        print("sources:")
        for r in ans.retrieved:
            url = r.chunk.metadata.get("url", "")
            title = r.chunk.metadata.get("title", r.chunk.source_id)
            print(f"  [{r.rank + 1}] {title}  {url}")
        print()


if __name__ == "__main__":
    main()
