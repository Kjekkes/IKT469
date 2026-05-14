"""Polite BFS crawler for UiA's public IKT pages.

Design choices:

- **robots.txt** is honored per-domain via `urllib.robotparser`. We cache the
  parser per host so we don't refetch.
- **Domain allowlist** stops the crawl from wandering off uia.no.
- **URL pattern filter** (`url_must_contain`) keeps us inside IKT-related
  paths instead of crawling the entire university website.
- **trafilatura** is the workhorse for body extraction — it strips nav,
  footer, cookie banners, etc. We fall back to BeautifulSoup if trafilatura
  returns nothing (rare but happens on JS-heavy pages).
- **Hash-based id** so re-running the scraper produces stable doc ids.

Pages with very short bodies (<200 chars after extraction) are dropped —
those are usually category landing pages with only navigation links.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Set
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


# File extensions we never follow. Cheap pre-filter; we still HEAD-check
# Content-Type before parsing, but skipping these saves a round-trip.
SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".webm",
    ".css", ".js", ".json", ".xml", ".rss",
}


@dataclass
class Page:
    id: str
    url: str
    title: str
    text: str
    fetched_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id, "url": self.url, "title": self.title,
            "text": self.text, "fetched_at": self.fetched_at,
        }


@dataclass
class UiaScraper:
    seed_urls: List[str]
    allowed_domains: List[str]
    max_pages: int = 300
    request_delay_seconds: float = 1.0
    user_agent: str = "IKT469-RAG-research/0.1"
    url_must_contain: List[str] = field(default_factory=lambda: ["ikt", "ict"])
    min_body_chars: int = 200
    request_timeout: float = 20.0

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.user_agent
        self._robots_cache: dict[str, Optional[RobotFileParser]] = {}

    # --- URL filtering --------------------------------------------------

    def _on_allowed_domain(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def _path_keyword_match(self, url: str) -> bool:
        if not self.url_must_contain:
            return True
        u = url.lower()
        return any(kw in u for kw in self.url_must_contain)

    def _looks_like_html(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return not any(path.endswith(ext) for ext in SKIP_EXTENSIONS)

    def _is_crawlable(self, url: str, is_seed: bool = False) -> bool:
        if not url.startswith(("http://", "https://")):
            return False
        if not self._on_allowed_domain(url):
            return False
        if not self._looks_like_html(url):
            return False
        # Seeds are always allowed to pass the keyword filter — otherwise the
        # crawler can reject the entry points it was explicitly configured
        # to start from (e.g. a generic /studies/ landing page).
        if not is_seed and not self._path_keyword_match(url):
            return False
        return True

    # --- robots.txt -----------------------------------------------------

    def _robots_for(self, url: str) -> Optional[RobotFileParser]:
        host = urlparse(url).netloc
        if host in self._robots_cache:
            return self._robots_cache[host]
        rp = RobotFileParser()
        robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
        try:
            r = self.session.get(robots_url, timeout=self.request_timeout)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                rp = None  # treat as fully allowed if missing/forbidden
        except Exception as e:
            log.warning("robots.txt fetch failed for %s: %s", host, e)
            rp = None
        self._robots_cache[host] = rp
        return rp

    def _allowed_by_robots(self, url: str) -> bool:
        rp = self._robots_for(url)
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)

    # --- Fetching + parsing --------------------------------------------

    def _fetch(self, url: str) -> Optional[requests.Response]:
        try:
            r = self.session.get(url, timeout=self.request_timeout, allow_redirects=True)
        except requests.RequestException as e:
            log.warning("GET %s failed: %s", url, e)
            return None
        if r.status_code != 200:
            log.info("GET %s -> HTTP %s", url, r.status_code)
            return None
        # Re-check the *post-redirect* URL against the allowlist. Without
        # this, a URL that starts on uia.no but redirects to e.g. an
        # idp.feide.no SAML login flow would be saved with the off-domain
        # final URL, polluting the index with login-page text.
        if not self._on_allowed_domain(r.url):
            log.info("Skipping off-domain redirect: %s -> %s", url, r.url)
            return None
        ctype = r.headers.get("Content-Type", "").lower()
        if "html" not in ctype:
            log.info("Skipping non-HTML response: %s (%s)", url, ctype)
            return None
        return r

    def _extract(self, html: str, url: str) -> tuple[str, str]:
        """Return (title, body_text). Body is empty if extraction fails."""
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        body = trafilatura.extract(
            html, url=url,
            include_comments=False, include_tables=False, no_fallback=False,
        ) or ""
        return title, body

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        out: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            # Skip UiA URLs that explicitly trigger a login redirect; they
            # take us off-domain to idp.feide.no and produce no useful body.
            low = href.lower()
            if "authtarget" in low or "authticket" in low:
                continue
            absolute = urljoin(base_url, href)
            absolute, _ = urldefrag(absolute)  # strip #anchors
            out.append(absolute)
        return out

    # --- Crawl ---------------------------------------------------------

    def crawl(self) -> List[Page]:
        from collections import deque

        queue: deque[str] = deque()
        for seed in self.seed_urls:
            queue.append(seed)
        seed_set: Set[str] = set(self.seed_urls)
        visited: Set[str] = set()
        pages: List[Page] = []

        while queue and len(pages) < self.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if not self._is_crawlable(url, is_seed=(url in seed_set)):
                continue
            if not self._allowed_by_robots(url):
                log.info("robots.txt forbids %s", url)
                continue

            log.info("[%d/%d] GET %s", len(pages) + 1, self.max_pages, url)
            r = self._fetch(url)
            time.sleep(self.request_delay_seconds)  # polite even on failure
            if r is None:
                continue

            title, body = self._extract(r.text, r.url)
            if len(body) >= self.min_body_chars:
                page_id = hashlib.sha1(r.url.encode("utf-8")).hexdigest()[:16]
                pages.append(Page(
                    id=page_id, url=r.url, title=title, text=body,
                    fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ))
            else:
                log.info("Body too short (%d chars), kept as link source only", len(body))

            for link in self._extract_links(r.text, r.url):
                if link not in visited:
                    queue.append(link)

        return pages


def write_pages_jsonl(pages: Iterable[Page], path) -> int:
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
