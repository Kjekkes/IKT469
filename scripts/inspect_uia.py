"""Inspect the scraped UiA corpus.

Three modes:

  python scripts/inspect_uia.py                    # list all pages: title, url, length, snippet
  python scripts/inspect_uia.py --search "master"  # show pages whose title/text contains the term
  python scripts/inspect_uia.py --show <id-prefix> # dump a single page in full

Use this *after* running scripts/03_scrape_uia.py to figure out which
pages actually got crawled, so you can write the UiA eval questions
against verified content.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import load_config


def iter_pages(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def list_pages(pages: list[dict], search: str | None) -> None:
    if search:
        s = search.lower()
        pages = [p for p in pages
                 if s in (p.get("title") or "").lower()
                 or s in (p.get("text") or "").lower()
                 or s in (p.get("url") or "").lower()]
    print(f"\n{len(pages)} page(s)\n")
    for p in pages:
        text = p.get("text") or ""
        snippet = text[:140].replace("\n", " ").strip()
        if len(text) > 140:
            snippet += "..."
        print(f"  id={p['id']}  chars={len(text):5d}")
        print(f"    title: {p.get('title') or '(no title)'}")
        print(f"    url:   {p.get('url')}")
        print(f"    text:  {snippet}")
        print()


def show_page(pages: list[dict], id_prefix: str) -> None:
    matches = [p for p in pages if p["id"].startswith(id_prefix)]
    if not matches:
        print(f"No page with id starting with {id_prefix!r}.")
        return
    if len(matches) > 1:
        print(f"Ambiguous prefix {id_prefix!r}, matches {len(matches)} pages:")
        for p in matches:
            print(f"  {p['id']}  {p.get('title')}")
        return
    p = matches[0]
    print(f"--- {p.get('title')} ---")
    print(f"url:        {p.get('url')}")
    print(f"id:         {p['id']}")
    print(f"fetched_at: {p.get('fetched_at')}")
    print(f"chars:      {len(p.get('text') or '')}")
    print()
    print(p.get("text") or "(empty)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", help="Filter pages by case-insensitive substring match")
    ap.add_argument("--show", help="Dump a single page in full (id prefix is enough)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")
    path = repo_root / config["paths"]["uia_clean_dir"] / "pages.jsonl"
    if not path.exists():
        print(f"No pages at {path}. Run scripts/03_scrape_uia.py first.")
        sys.exit(1)
    pages = list(iter_pages(path))

    if args.show:
        show_page(pages, args.show)
    else:
        list_pages(pages, args.search)


if __name__ == "__main__":
    main()
