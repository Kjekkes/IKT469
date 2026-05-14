"""Scrape UiA IKT pages and save cleaned text.

Output:
    data/uia_clean/pages.jsonl    one JSON object per page

Run:
    python scripts/03_scrape_uia.py

Tune the URL allowlist (`uia.url_must_contain`) and `max_pages` in
config.yaml. The defaults stay inside IKT-related paths and cap the crawl
at 300 pages — a few minutes on a normal connection at 1 req/s.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import load_config
from src.uia_scrape import UiaScraper, write_pages_jsonl


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")
    cfg = config["uia"]

    out_path = repo_root / config["paths"]["uia_clean_dir"] / "pages.jsonl"

    scraper_kwargs = dict(
        seed_urls=cfg["seed_urls"],
        allowed_domains=cfg["allowed_domains"],
        max_pages=cfg["max_pages"],
        request_delay_seconds=cfg["request_delay_seconds"],
        user_agent=cfg["user_agent"],
    )
    # Optional override; falls back to the dataclass default if absent.
    if cfg.get("url_must_contain") is not None:
        scraper_kwargs["url_must_contain"] = cfg["url_must_contain"]
    scraper = UiaScraper(**scraper_kwargs)

    print(f"Crawling from {cfg['seed_urls']} (max {cfg['max_pages']} pages)...")
    t0 = time.time()
    pages = scraper.crawl()
    n_written = write_pages_jsonl(pages, out_path)
    print(f"\nWrote {n_written} pages to {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
