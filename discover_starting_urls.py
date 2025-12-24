#!/usr/bin/env python3
"""
discover_starting_urls.py

Stage 1 (Search API only): discover "starting" URLs for later crawling.

INPUT CSV header (exact):
  publisher,rt_publisher_url,authors

- publisher: publication name (string)
- rt_publisher_url: use as publication_id in output (string)
- authors: optional; if present, treated as "publication has authors"
          delimiter recommended: ';' or '|' (commas can break "Last, First")

OUTPUT CSV header (exact; 5 columns):
  publication_id,publication,author,source_url,source_url_type

Where source_url_type ∈:
  - publication_review
  - publication_home
  - publication_with_author

Rules:
- If authors is empty:
    emit up to 2 rows:
      (publication_review) best review index page we can find
      (publication_home)   best homepage/official site we can find
- If authors is present:
    for each author, emit 1 row:
      (publication_with_author) best author page inside that publication

Speed features for GitHub Actions:
- Sharding:
    --shard-index i --shard-count n
    processes only rows where (row_number % n == i)
- Incremental skip:
    --existing merged_starting_urls.csv
    skips (publication_id, author, source_url_type) already present

Brave Search API:
- set env var BRAVE_API_KEY
- uses Brave Web Search endpoint

Usage examples:
  export BRAVE_API_KEY="..."
  python discover_starting_urls.py \
    --input data/publications_input.csv \
    --output outputs/starting_urls_0.csv \
    --existing outputs/merged_starting_urls.csv \
    --shard-index 0 --shard-count 8 \
    --sleep 0.2
"""

import argparse
import csv
import os
import re
import time
import urllib.parse
from typing import Dict, List, Optional, Set, Tuple

import requests

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


# ----------------------------
# Utility
# ----------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def safe_slug(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def parse_authors(authors_raw: str) -> List[str]:
    """
    Prefer ';' or '|' as delimiters.
    Comma split is a last resort (may break 'Last, First').
    """
    s = normalize_ws(authors_raw)
    if not s:
        return []
    if ";" in s:
        parts = [normalize_ws(p) for p in s.split(";")]
    elif "|" in s:
        parts = [normalize_ws(p) for p in s.split("|")]
    else:
        parts = [normalize_ws(p) for p in s.split(",")]
    # Deduplicate while keeping order
    seen = set()
    out = []
    for p in parts:
        if p and p.lower() not in seen:
            out.append(p)
            seen.add(p.lower())
    return out


def hostname(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def path_depth(url: str) -> int:
    try:
        p = urllib.parse.urlparse(url).path
        return len([x for x in p.split("/") if x])
    except Exception:
        return 999


def is_social_or_wiki(url: str) -> bool:
    u = (url or "").lower()
    return any(d in u for d in [
        "twitter.com", "x.com", "facebook.com", "instagram.com",
        "linkedin.com", "wikipedia.org", "imdb.com"
    ])


# ----------------------------
# Brave Search
# ----------------------------

def brave_search(api_key: str, query: str, count: int = 8) -> List[Dict]:
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": "RT-StartingURLDiscovery/1.0",
    }
    params = {"q": query, "count": count}
    r = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    return data.get("web", {}).get("results", []) or []


# ----------------------------
# Scoring heuristics
# ----------------------------

def score_publication_review_candidate(url: str, title: str, desc: str) -> int:
    """
    Prefer review index/section pages.
    """
    u = (url or "").lower()
    t = (title or "").lower()
    d = (desc or "").lower()
    text = f"{u} {t} {d}"

    score = 0

    # Strong review-index signals
    if re.search(r"/reviews?\b", u):
        score += 10
    if any(k in text for k in ["movie reviews", "film reviews", "reviews archive", "review archive"]):
        score += 10
    if any(k in u for k in ["/movies/reviews", "/film/reviews", "/cinema/reviews"]):
        score += 8

    # Medium signals: movie/film sections
    if any(k in u for k in ["/movies", "/film", "/cinema", "/entertainment"]):
        score += 4
    if any(k in text for k in ["movies", "film", "cinema", "entertainment"]):
        score += 2

    # Prefer section/index pages (shallower)
    depth = path_depth(url)
    if depth <= 2:
        score += 3
    elif depth >= 6:
        score -= 3

    # Penalize obvious non-content pages
    if any(k in u for k in ["/about", "/privacy", "/terms", "/contact", "/subscribe", "/newsletter", "/donate"]):
        score -= 12

    # Penalize tag/category pages slightly (often noisy, but sometimes OK)
    if any(k in u for k in ["/tag/", "/tags/", "/category/", "/topics/"]):
        score -= 2

    # Penalize social / wiki
    if is_social_or_wiki(u):
        score -= 25

    return score


def score_publication_home_candidate(url: str, title: str, desc: str) -> int:
    """
    Prefer the official homepage domain and root-ish URLs, avoid aggregators.
    """
    u = (url or "").lower()
    score = 0

    # Root/home is good
    depth = path_depth(url)
    if depth == 0:
        score += 10
    elif depth == 1:
        score += 6
    elif depth >= 5:
        score -= 4

    # Avoid obvious junk
    if any(k in u for k in ["/about", "/privacy", "/terms", "/contact"]):
        score -= 10

    # Avoid social/wiki
    if is_social_or_wiki(u):
        score -= 25

    # Prefer https
    if u.startswith("https://"):
        score += 1

    return score


def score_author_candidate(url: str, author: str, publisher: str, title: str, desc: str) -> int:
    """
    Prefer author pages on publisher domain (best effort).
    """
    u = (url or "").lower()
    t = (title or "").lower()
    d = (desc or "").lower()
    text = f"{u} {t} {d}"

    score = 0
    a_slug = safe_slug(author)

    # Common author URL patterns
    if any(k in u for k in ["/author/", "/authors/", "/by/", "/staff/", "/contributors/"]):
        score += 12

    # Author slug in URL is strong
    if a_slug and a_slug in u:
        score += 10

    # Title/desc contains author name
    if author.lower() in (title or "").lower():
        score += 5
    if author.lower() in (desc or "").lower():
        score += 3

    # Penalize social/wiki
    if is_social_or_wiki(u):
        score -= 30

    # Prefer shallower author pages, penalize deep article pages
    depth = path_depth(url)
    if depth <= 3:
        score += 3
    elif depth >= 7:
        score -= 5

    # Weak boost if publisher name appears (not always reliable)
    if publisher.lower() in text:
        score += 1

    return score


def pick_best(results: List[Dict], scorer_fn) -> Optional[str]:
    best_url = None
    best_score = -10**9
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        title = r.get("title") or ""
        desc = r.get("description") or ""
        s = scorer_fn(url, title, desc)
        if s > best_score:
            best_score = s
            best_url = url
    return best_url


# ----------------------------
# Discovery routines
# ----------------------------

def discover_publication_review_url(api_key: str, publisher: str, per_query: int = 8) -> Optional[str]:
    # High-signal queries first. Keep this short for speed.
    queries = [
        f'"{publisher}" "movie reviews"',
        f'"{publisher}" "film reviews"',
        f'"{publisher}" reviews movies',
    ]

    results: List[Dict] = []
    for q in queries:
        try:
            results.extend(brave_search(api_key, q, count=per_query))
        except requests.HTTPError:
            # Bubble up (caller may want to backoff), but keep robust for single failures.
            continue
        except Exception:
            continue

    if not results:
        return None

    best = None
    best_score = -10**9
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        s = score_publication_review_candidate(url, r.get("title", ""), r.get("description", ""))
        if s > best_score:
            best_score = s
            best = url

    return best


def discover_publication_home_url(api_key: str, publisher: str, per_query: int = 8) -> Optional[str]:
    queries = [
        f'"{publisher}" official site',
        f'"{publisher}" homepage',
    ]

    results: List[Dict] = []
    for q in queries:
        try:
            results.extend(brave_search(api_key, q, count=per_query))
        except Exception:
            continue

    if not results:
        return None

    best = None
    best_score = -10**9
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        s = score_publication_home_candidate(url, r.get("title", ""), r.get("description", ""))
        if s > best_score:
            best_score = s
            best = url
    return best


def discover_author_url(api_key: str, publisher: str, author: str, per_query: int = 8) -> Optional[str]:
    # A few targeted templates. Keep short for speed.
    queries = [
        f'"{author}" "{publisher}" author',
        f'"{author}" "{publisher}" "/author/"',
        f'"{author}" "{publisher}" "/by/"',
        f'"{author}" "{publisher}" film review',
    ]

    results: List[Dict] = []
    for q in queries:
        try:
            results.extend(brave_search(api_key, q, count=per_query))
        except Exception:
            continue

    if not results:
        return None

    best = None
    best_score = -10**9
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        s = score_author_candidate(url, author, publisher, r.get("title", ""), r.get("description", ""))
        if s > best_score:
            best_score = s
            best = url

    return best


# ----------------------------
# Incremental skip
# ----------------------------

def load_existing_keys(existing_csv_path: Optional[str]) -> Set[Tuple[str, str, str]]:
    """
    Returns keys for skipping already discovered items:
      (publication_id, author_lower_or_empty, source_url_type)
    """
    keys: Set[Tuple[str, str, str]] = set()
    if not existing_csv_path:
        return keys
    if not os.path.exists(existing_csv_path):
        return keys

    with open(existing_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"publication_id", "author", "source_url_type"}
        if not reader.fieldnames or not needed.issubset(set(reader.fieldnames)):
            return keys
        for row in reader:
            pub_id = normalize_ws(row.get("publication_id", ""))
            author = normalize_ws(row.get("author", "")).lower()
            typ = normalize_ws(row.get("source_url_type", ""))
            if pub_id and typ:
                keys.add((pub_id, author, typ))
    return keys


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV: publisher,rt_publisher_url,authors")
    ap.add_argument("--output", required=True, help="Output CSV with 5 columns")
    ap.add_argument("--existing", default="", help="Existing merged output to skip already done work")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between API calls (per row/author)")
    ap.add_argument("--shard-index", type=int, default=0, help="Shard index (0..shard-count-1)")
    ap.add_argument("--shard-count", type=int, default=1, help="Number of shards")
    ap.add_argument("--per-query", type=int, default=8, help="Brave results per query")
    args = ap.parse_args()

    api_key = (os.environ.get("BRAVE_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing BRAVE_API_KEY env var.")

    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("Invalid shard args: require 0 <= shard-index < shard-count and shard-count >= 1")

    existing_keys = load_existing_keys(args.existing)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"publisher", "rt_publisher_url", "authors"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"Input must have columns exactly including: {sorted(required)}. Found: {reader.fieldnames}")
        in_rows = list(reader)

    # Write output
    out_fields = ["publication_id", "publication", "author", "source_url", "source_url_type"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()

        for idx, row in enumerate(in_rows):
            if (idx % args.shard_count) != args.shard_index:
                continue

            publisher = normalize_ws(row.get("publisher", ""))
            pub_id = normalize_ws(row.get("rt_publisher_url", ""))  # publication_id output
            authors_raw = row.get("authors", "") or ""
            authors = parse_authors(authors_raw)

            if not publisher or not pub_id:
                continue

            # Author-scoped publication
            if authors:
                for author in authors:
                    skip_key = (pub_id, author.lower(), "publication_with_author")
                    if skip_key in existing_keys:
                        continue

                    url = discover_author_url(api_key, publisher, author, per_query=args.per_query)
                    if url:
                        w.writerow({
                            "publication_id": pub_id,
                            "publication": publisher,
                            "author": author,
                            "source_url": url,
                            "source_url_type": "publication_with_author",
                        })
                        existing_keys.add(skip_key)

                    time.sleep(args.sleep)

            # Publication-scoped publication
            else:
                # Try review index first
                review_key = (pub_id, "", "publication_review")
                if review_key not in existing_keys:
                    review_url = discover_publication_review_url(api_key, publisher, per_query=args.per_query)
                    if review_url:
                        w.writerow({
                            "publication_id": pub_id,
                            "publication": publisher,
                            "author": "",
                            "source_url": review_url,
                            "source_url_type": "publication_review",
                        })
                        existing_keys.add(review_key)
                    time.sleep(args.sleep)

                # Then home page fallback (still useful even if review exists)
                home_key = (pub_id, "", "publication_home")
                if home_key not in existing_keys:
                    home_url = discover_publication_home_url(api_key, publisher, per_query=args.per_query)
                    if home_url:
                        w.writerow({
                            "publication_id": pub_id,
                            "publication": publisher,
                            "author": "",
                            "source_url": home_url,
                            "source_url_type": "publication_home",
                        })
                        existing_keys.add(home_key)
                    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
