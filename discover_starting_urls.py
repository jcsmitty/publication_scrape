#!/usr/bin/env python3
"""
discover_starting_urls.py  (TWO-PHASE, OFFICIAL-DOMAIN-FIRST)

Two-phase discovery (Search API only, no crawling):
PHASE 1: Find the publication's official HOME domain (publication_home).
PHASE 2: Constrain subsequent discovery to that domain using `site:...`
         to find publication_review (or best section page) and author pages.

INPUT CSV header (exact):
  publisher,rt_publisher_url,authors

OUTPUT CSV header (exact; 5 columns):
  publication_id,publication,author,source_url,source_url_type

Where:
- publication_id = rt_publisher_url
- publication     = publisher
- author          = author (blank if none)
- source_url_type ∈ { publication_home, publication_review, publication_with_author }

Hard constraints:
- source_url must NOT be on Rotten Tomatoes or other blocked domains.
- Prefer domains that match publication tokens.

Speed features (GitHub Actions):
- --shard-index / --shard-count
- --existing merged_starting_urls.csv  (skip already-discovered keys)
- Keep --sleep small (0.0–0.3)

Env var:
- BRAVE_API_KEY

Usage:
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

# Domains we never want as "source" pages.
BLOCKED_DOMAINS = {
    "rottentomatoes.com", "www.rottentomatoes.com",
    "metacritic.com", "www.metacritic.com",
    "imdb.com", "www.imdb.com",
    "letterboxd.com", "www.letterboxd.com",
    "wikipedia.org", "www.wikipedia.org",
    "twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com",
    "youtube.com", "www.youtube.com",
    "tiktok.com", "www.tiktok.com",
    "reddit.com", "www.reddit.com",
    "medium.com",
}


# ----------------------------
# Utility
# ----------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def safe_slug(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def parse_authors(authors_raw: str) -> List[str]:
    """
    Prefer ';' or '|' delimiters.
    Comma split is last resort (breaks "Last, First").
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
    seen = set()
    out = []
    for p in parts:
        k = p.lower()
        if p and k not in seen:
            out.append(p)
            seen.add(k)
    return out


def canonicalize_url(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if not url:
        return None
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https"):
        return None

    # Strip fragment and common tracking params
    q = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
    q2 = [(k, v) for (k, v) in q if not re.match(r"^(utm_|fbclid|gclid|mc_cid|mc_eid)", k, re.I)]
    new_query = urllib.parse.urlencode(q2)
    p2 = p._replace(query=new_query, fragment="")
    return p2.geturl()


def get_domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return ""


def is_blocked(url: str) -> bool:
    d = get_domain(url)
    if not d:
        return True
    if d.startswith("www."):
        d2 = d[4:]
    else:
        d2 = d
    if d in BLOCKED_DOMAINS or d2 in BLOCKED_DOMAINS:
        return True
    for bd in BLOCKED_DOMAINS:
        if d.endswith("." + bd) or d2.endswith("." + bd):
            return True
    return False


def path_depth(url: str) -> int:
    try:
        p = urllib.parse.urlparse(url).path
        return len([x for x in p.split("/") if x])
    except Exception:
        return 999


def root_url(url: str) -> Optional[str]:
    try:
        p = urllib.parse.urlparse(url)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}/"
    except Exception:
        pass
    return None


def core_domain(netloc: str) -> str:
    """
    Approximate core domain extractor (no external deps).
    - www.nytimes.com -> nytimes.com
    - foo.guardian.co.uk -> guardian.co.uk
    """
    netloc = (netloc or "").lower().split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    parts = netloc.split(".")
    if len(parts) <= 2:
        return netloc
    sld = {"co", "com", "org", "net", "gov", "edu"}
    if parts[-2] in sld and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def pub_tokens(publisher: str) -> List[str]:
    """
    Tokens for domain matching.
    Drops stopwords and short tokens.
    """
    s = (publisher or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t]
    stop = {"the", "a", "an", "and", "of", "for", "to", "at", "on", "in"}
    tokens = [t for t in tokens if t not in stop and len(t) >= 4]
    joined = "".join(tokens)
    out = list(dict.fromkeys(tokens + ([joined] if len(joined) >= 6 else [])))
    return out


def domain_affinity_score(url: str, publisher: str) -> int:
    """
    Boost if core domain contains publisher tokens.
    """
    d = core_domain(get_domain(url))
    toks = pub_tokens(publisher)
    score = 0
    for t in toks:
        if t and t in d:
            score += 18
    for t in toks:
        if d.startswith(t + ".") or d.startswith(t):
            score += 6
    # Penalize common hosting platforms that are rarely "official" for pubs
    if any(x in d for x in ["blogspot.", "wordpress.", "substack.com", "medium.com"]):
        score -= 12
    return score


# ----------------------------
# Brave Search
# ----------------------------

def brave_search(api_key: str, query: str, count: int = 8) -> List[Dict]:
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": "RT-StartingURLDiscovery/2.0",
    }
    params = {"q": query, "count": count}
    r = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    return data.get("web", {}).get("results", []) or []


def filter_results(results: List[Dict]) -> List[Dict]:
    out = []
    for r in results:
        u = canonicalize_url(r.get("url") or "")
        if not u:
            continue
        if is_blocked(u):
            continue
        r2 = dict(r)
        r2["url"] = u
        out.append(r2)
    return out


# ----------------------------
# Scoring
# ----------------------------

def score_home_candidate(url: str, title: str, desc: str, publisher: str) -> int:
    """
    Score for PHASE 1: official home domain.
    """
    u = (url or "").lower()
    score = 0

    score += domain_affinity_score(url, publisher)  # official-domain boost

    # Prefer root-ish URLs
    depth = path_depth(url)
    if depth == 0:
        score += 14
    elif depth == 1:
        score += 9
    elif depth >= 5:
        score -= 5

    # Penalize non-home pages
    if any(k in u for k in ["/about", "/privacy", "/terms", "/contact", "/subscribe", "/newsletter", "/donate"]):
        score -= 10

    # Prefer https
    if u.startswith("https://"):
        score += 1

    return score


def score_review_candidate(url: str, title: str, desc: str, publisher: str, target_core: Optional[str]) -> int:
    """
    Score for PHASE 2: review/section page *within official domain if possible*.
    """
    u = (url or "").lower()
    t = (title or "").lower()
    d = (desc or "").lower()
    text = f"{u} {t} {d}"

    score = 0
    score += domain_affinity_score(url, publisher)

    # Hard boost if within the target core domain
    if target_core:
        if core_domain(get_domain(url)) == target_core:
            score += 35
        else:
            score -= 10

    # Strong review signals
    if re.search(r"/reviews?\b", u):
        score += 14
    if any(k in text for k in ["movie reviews", "film reviews", "reviews archive", "review archive"]):
        score += 10
    if any(k in u for k in ["/movie-reviews", "/film-reviews", "/reviews/movies", "/reviews/film"]):
        score += 8

    # Medium: entertainment/movies sections
    if any(k in u for k in ["/movies", "/film", "/cinema", "/entertainment"]):
        score += 4

    # Prefer index pages
    depth = path_depth(url)
    if depth <= 2:
        score += 3
    elif depth >= 7:
        score -= 4

    # Penalize non-content
    if any(k in u for k in ["/about", "/privacy", "/terms", "/contact", "/subscribe", "/newsletter", "/donate"]):
        score -= 15
    if any(k in u for k in ["/tag/", "/tags/", "/category/", "/topics/"]):
        score -= 3

    return score


def score_author_candidate(url: str, author: str, publisher: str, title: str, desc: str,
                           target_core: Optional[str]) -> int:
    u = (url or "").lower()
    t = (title or "").lower()
    d = (desc or "").lower()
    text = f"{u} {t} {d}"

    score = 0
    score += domain_affinity_score(url, publisher)

    # Prefer official domain
    if target_core:
        if core_domain(get_domain(url)) == target_core:
            score += 35
        else:
            score -= 10

    a_slug = safe_slug(author)

    if any(k in u for k in ["/author/", "/authors/", "/by/", "/staff/", "/contributors/"]):
        score += 16
    if a_slug and a_slug in u:
        score += 10
    if author.lower() in t:
        score += 4
    if author.lower() in d:
        score += 2
    if publisher.lower() in text:
        score += 1

    depth = path_depth(url)
    if depth <= 3:
        score += 3
    elif depth >= 8:
        score -= 6

    return score


# ----------------------------
# Two-phase discovery
# ----------------------------

def discover_home(api_key: str, publisher: str, per_query: int = 8) -> Optional[str]:
    queries = [
        f'"{publisher}" official site',
        f'"{publisher}" homepage',
        f'"{publisher}" website',
    ]
    results: List[Dict] = []
    for q in queries:
        try:
            results.extend(brave_search(api_key, q, count=per_query))
        except Exception:
            continue

    results = filter_results(results)
    if not results:
        return None

    best_url = None
    best_score = -10**9
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        s = score_home_candidate(url, r.get("title", ""), r.get("description", ""), publisher)
        if s > best_score:
            best_score = s
            best_url = url

    if best_url:
        return root_url(best_url) or best_url
    return None


def discover_review_within_domain(api_key: str, publisher: str, home_url: Optional[str],
                                 per_query: int = 8) -> Optional[str]:
    target_core = core_domain(get_domain(home_url)) if home_url else None
    site_clause = f"site:{target_core}" if target_core else ""

    queries = [
        f'{site_clause} "{publisher}" "movie reviews"',
        f'{site_clause} "{publisher}" "film reviews"',
        f"{site_clause} reviews movies",
        f"{site_clause} film reviews",
        f"{site_clause} movie reviews",
    ]

    results: List[Dict] = []
    for q in queries:
        qn = normalize_ws(q)
        if not qn:
            continue
        try:
            results.extend(brave_search(api_key, qn, count=per_query))
        except Exception:
            continue

    results = filter_results(results)
    if not results:
        return None

    best_url = None
    best_score = -10**9
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        s = score_review_candidate(url, r.get("title", ""), r.get("description", ""), publisher, target_core)
        if s > best_score:
            best_score = s
            best_url = url

    return best_url


def discover_author_within_domain(api_key: str, publisher: str, author: str, home_url: Optional[str],
                                 per_query: int = 8) -> Optional[str]:
    target_core = core_domain(get_domain(home_url)) if home_url else None
    site_clause = f"site:{target_core}" if target_core else ""

    queries = [
        f'{site_clause} "{author}" author',
        f'{site_clause} "{author}" "/author/"',
        f'{site_clause} "{author}" "/by/"',
        f'{site_clause} "{author}" staff',
        f'{site_clause} "{author}" contributor',
    ]

    results: List[Dict] = []
    for q in queries:
        qn = normalize_ws(q)
        if not qn:
            continue
        try:
            results.extend(brave_search(api_key, qn, count=per_query))
        except Exception:
            continue

    results = filter_results(results)
    if not results:
        return None

    best_url = None
    best_score = -10**9
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        s = score_author_candidate(url, author, publisher, r.get("title", ""), r.get("description", ""), target_core)
        if s > best_score:
            best_score = s
            best_url = url

    return best_url


# ----------------------------
# Incremental skip
# ----------------------------

def load_existing_keys(existing_csv_path: Optional[str]) -> Set[Tuple[str, str, str]]:
    """
    Skip keys:
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
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between API calls")
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
            raise SystemExit(f"Input must have columns including: {sorted(required)}. Found: {reader.fieldnames}")
        in_rows = list(reader)

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

            # Phase 1: find official home
            home_key = (pub_id, "", "publication_home")
            home_url: Optional[str] = None
            if home_key not in existing_keys:
                home_url = discover_home(api_key, publisher, per_query=args.per_query)
                if home_url and not is_blocked(home_url):
                    w.writerow({
                        "publication_id": pub_id,
                        "publication": publisher,
                        "author": "",
                        "source_url": home_url,
                        "source_url_type": "publication_home",
                    })
                    existing_keys.add(home_key)
                time.sleep(args.sleep)
            else:
                # If we skipped writing home, still try to recover a home_url for phase 2 by searching lightly.
                # (Avoids needing to read existing file for the actual URL.)
                home_url = discover_home(api_key, publisher, per_query=max(4, args.per_query // 2))
                time.sleep(args.sleep)

            # Author-scoped publications
            if authors:
                for author in authors:
                    akey = (pub_id, author.lower(), "publication_with_author")
                    if akey in existing_keys:
                        continue

                    aurl = discover_author_within_domain(api_key, publisher, author, home_url, per_query=args.per_query)
                    if aurl and not is_blocked(aurl):
                        w.writerow({
                            "publication_id": pub_id,
                            "publication": publisher,
                            "author": author,
                            "source_url": aurl,
                            "source_url_type": "publication_with_author",
                        })
                        existing_keys.add(akey)
                    time.sleep(args.sleep)

            # Publication-scoped publications: also find a review/section page
            else:
                rkey = (pub_id, "", "publication_review")
                if rkey in existing_keys:
                    continue
                rurl = discover_review_within_domain(api_key, publisher, home_url, per_query=args.per_query)
                if rurl and not is_blocked(rurl):
                    w.writerow({
                        "publication_id": pub_id,
                        "publication": publisher,
                        "author": "",
                        "source_url": rurl,
                        "source_url_type": "publication_review",
                    })
                    existing_keys.add(rkey)
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
