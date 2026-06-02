"""
ArXiv Producer
==============
Collecte des articles scientifiques depuis l'API ArXiv (Atom/XML)
et les publie sur le topic Kafka `arxiv-articles`.

Usage:
    python producers/arxiv_producer.py
    python producers/arxiv_producer.py --query "graph neural network" --max 200
    python producers/arxiv_producer.py --category cs.LG --max 100 --loop --interval 60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests
from kafka import KafkaProducer
from kafka.errors import KafkaError

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    KAFKA_BOOTSTRAP_SERVERS, TOPIC_ARXIV, TOPIC_DLQ,
    ARXIV_BASE_URL, ARXIV_BATCH_SIZE,
    ARXIV_DEFAULT_QUERY, ARXIV_CATEGORIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ArXiv] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arxiv_producer")

# ArXiv Atom namespace
NS = {
    "atom":   "http://www.w3.org/2005/Atom",
    "arxiv":  "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_arxiv_articles(query: str, start: int = 0, max_results: int = 50) -> list[dict]:
    """Fetch articles from ArXiv API. Returns list of article dicts."""
    # If query already contains a field prefix (cat:, ti:, au:, abs:), use as-is
    # Otherwise wrap with all: for full-text search
    if ":" in query.split()[0]:
        search_query = query
    else:
        search_query = f"all:{query}"

    params = {
        "search_query": search_query,
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    log.info(f"Fetching ArXiv: '{query}' offset={start} max={max_results}")
    resp = requests.get(ARXIV_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    root    = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", NS)
    log.info(f"Retrieved {len(entries)} entries")

    articles = []
    for entry in entries:
        try:
            # ID → arxiv_id
            raw_id   = entry.findtext("atom:id", "", NS)
            arxiv_id = raw_id.split("/abs/")[-1].replace("/", "_") if "/abs/" in raw_id else raw_id

            # Title (strip newlines)
            title = " ".join((entry.findtext("atom:title", "", NS) or "").split())

            # Abstract
            summary = " ".join((entry.findtext("atom:summary", "", NS) or "").split())

            # Authors
            authors = [
                a.findtext("atom:name", "", NS)
                for a in entry.findall("atom:author", NS)
            ]

            # Published date
            published = entry.findtext("atom:published", "", NS)[:10]   # YYYY-MM-DD

            # Categories
            categories = [
                tag.get("term", "")
                for tag in entry.findall("atom:category", NS)
            ]
            primary_cat = categories[0] if categories else ""

            # PDF link
            pdf_url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            # DOI (if available)
            doi = entry.findtext("arxiv:doi", "", NS)

            # Journal ref
            journal_ref = entry.findtext("arxiv:journal_ref", "", NS)

            articles.append({
                "source":        "arxiv",
                "arxiv_id":      arxiv_id,
                "doi":           doi,
                "title":         title,
                "abstract":      summary[:2000],
                "authors":       authors,
                "categories":    categories,
                "primary_category": primary_cat,
                "journal_ref":   journal_ref,
                "pdf_url":       pdf_url,
                "published_at":  published,
                "ingested_at":   datetime.utcnow().isoformat(),
            })

        except Exception as e:
            log.warning(f"Failed to parse entry: {e}")
            continue

    return articles


def fetch_by_category(categories: list[str], max_per_cat: int) -> list[dict]:
    """Fetch articles for multiple ArXiv categories."""
    all_articles = []
    for cat in categories:
        try:
            articles = fetch_arxiv_articles(
                query=f"cat:{cat}",
                max_results=max_per_cat,
            )
            all_articles.extend(articles)
            time.sleep(3)   # ArXiv rate limit: 1 req/3s
        except Exception as e:
            log.error(f"Failed to fetch category {cat}: {e}")
    return all_articles


# ── Kafka producer ────────────────────────────────────────────────────────────

def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        acks="all",
        retries=3,
        max_block_ms=15_000,
    )


def publish_articles(producer: KafkaProducer, articles: list[dict]) -> int:
    """Publish articles to Kafka. Returns number successfully sent."""
    sent = 0
    seen_ids = set()
    for article in articles:
        article_id = article.get("arxiv_id", "")
        if article_id in seen_ids:
            continue
        seen_ids.add(article_id)
        try:
            future = producer.send(
                TOPIC_ARXIV,
                value=article,
                key=article_id.encode() if article_id else None,
            )
            future.get(timeout=10)
            sent += 1
            log.info(f"  ✓ Sent arXiv:{article_id} | {article['title'][:60]}…")
        except KafkaError as e:
            log.error(f"  ✗ Failed arXiv:{article_id}: {e}")
            try:
                producer.send(TOPIC_DLQ, value={"error": str(e), "article": article})
            except Exception:
                pass
    return sent


# ── Main ──────────────────────────────────────────────────────────────────────

def run(query: str | None, categories: list[str] | None,
        max_results: int, interval: int, loop: bool) -> None:
    log.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}…")
    producer = build_producer()
    log.info(f"Connected. Topic: {TOPIC_ARXIV}")

    total_sent = 0
    iteration  = 0

    while True:
        iteration += 1
        log.info(f"\n{'='*50}")
        log.info(f"Iteration {iteration} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        try:
            if categories:
                articles = fetch_by_category(categories, max_per_cat=max_results // len(categories))
            else:
                articles = []
                for offset in range(0, max_results, ARXIV_BATCH_SIZE):
                    batch = fetch_arxiv_articles(
                        query=query or ARXIV_DEFAULT_QUERY,
                        start=offset,
                        max_results=min(ARXIV_BATCH_SIZE, max_results - offset),
                    )
                    articles.extend(batch)
                    time.sleep(3)

            sent = publish_articles(producer, articles)
            total_sent += sent
            log.info(f"Iteration done: {sent}/{len(articles)} sent | Total: {total_sent}")

        except requests.RequestException as e:
            log.error(f"API error: {e}")
        except Exception as e:
            log.exception(f"Unexpected error: {e}")

        producer.flush()
        log.info(f"Total published so far: {total_sent} articles")

        if not loop:
            break
        log.info(f"Waiting {interval}s before next batch…")
        time.sleep(interval)

    producer.close()
    log.info(f"Producer stopped. Total sent: {total_sent}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ArXiv → Kafka producer")
    p.add_argument("--query",    default=None,           help="ArXiv search query")
    p.add_argument("--category", default=None, nargs="+",help="ArXiv categories (e.g. cs.LG cs.AI)")
    p.add_argument("--max",      type=int, default=100,  help="Max articles to fetch")
    p.add_argument("--interval", type=int, default=60,   help="Seconds between loops")
    p.add_argument("--loop",     action="store_true",    help="Run continuously")
    args = p.parse_args()

    run(
        query=args.query,
        categories=args.category or ARXIV_CATEGORIES,
        max_results=args.max,
        interval=args.interval,
        loop=args.loop,
    )
