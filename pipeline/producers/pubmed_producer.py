"""
PubMed Producer
===============
Collecte des articles scientifiques depuis l'API NCBI Entrez (PubMed)
et les publie sur le topic Kafka `pubmed-articles`.

Usage:
    python producers/pubmed_producer.py
    python producers/pubmed_producer.py --query "deep learning" --max 200
    python producers/pubmed_producer.py --query "graph neural network" --max 100 --interval 30
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
    KAFKA_BOOTSTRAP_SERVERS, TOPIC_PUBMED, TOPIC_DLQ,
    PUBMED_BASE_URL, PUBMED_EMAIL, PUBMED_TOOL,
    PUBMED_BATCH_SIZE, PUBMED_DEFAULT_QUERY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PubMed] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pubmed_producer")


# ── API helpers ───────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    log.info(f"Searching PubMed: '{query}' (max={max_results})")
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": PUBMED_EMAIL,
        "tool": PUBMED_TOOL,
    }
    resp = requests.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    pmids = resp.json()["esearchresult"]["idlist"]
    log.info(f"Found {len(pmids)} PMIDs")
    return pmids


def fetch_pubmed_articles(pmids: list[str]) -> list[dict]:
    """Fetch full article metadata for a batch of PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": PUBMED_EMAIL,
        "tool": PUBMED_TOOL,
    }
    resp = requests.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    articles = []

    for article_node in root.findall(".//PubmedArticle"):
        try:
            medline = article_node.find("MedlineCitation")
            article  = medline.find("Article")

            # PMID
            pmid = medline.findtext("PMID", "")

            # Title
            title = article.findtext("ArticleTitle", "").strip()

            # Abstract
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (p.get("Label", "") + ": " if p.get("Label") else "") + (p.text or "")
                for p in abstract_parts
            ).strip()

            # Authors
            authors = []
            for author in article.findall(".//Author"):
                last  = author.findtext("LastName", "")
                first = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last}, {first}".strip(", "))

            # Journal
            journal = article.findtext(".//Journal/Title", "")

            # Publication date
            pub_date = medline.find(".//DateCompleted") or medline.find(".//DateRevised")
            if pub_date is not None:
                year  = pub_date.findtext("Year", "")
                month = pub_date.findtext("Month", "01")
                day   = pub_date.findtext("Day", "01")
                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                date_str = ""

            # Keywords
            keywords = [kw.text for kw in article_node.findall(".//Keyword") if kw.text]

            # DOI
            doi = ""
            for id_node in article_node.findall(".//ArticleId"):
                if id_node.get("IdType") == "doi":
                    doi = id_node.text or ""
                    break

            articles.append({
                "source":        "pubmed",
                "pmid":          pmid,
                "doi":           doi,
                "title":         title,
                "abstract":      abstract[:2000],   # truncate for Kafka
                "authors":       authors,
                "journal":       journal,
                "keywords":      keywords,
                "published_at":  date_str,
                "ingested_at":   datetime.utcnow().isoformat(),
            })
        except Exception as e:
            log.warning(f"Failed to parse article: {e}")
            continue

    return articles


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
    for article in articles:
        try:
            future = producer.send(TOPIC_PUBMED, value=article, key=article["pmid"].encode())
            future.get(timeout=10)
            sent += 1
            log.info(f"  ✓ Sent PMID={article['pmid']} | {article['title'][:60]}…")
        except KafkaError as e:
            log.error(f"  ✗ Failed PMID={article['pmid']}: {e}")
            try:
                producer.send(TOPIC_DLQ, value={"error": str(e), "article": article})
            except Exception:
                pass
    return sent


# ── Main ──────────────────────────────────────────────────────────────────────

def run(query: str, max_results: int, interval: int, loop: bool) -> None:
    log.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}…")
    producer = build_producer()
    log.info(f"Connected. Topic: {TOPIC_PUBMED}")

    total_sent = 0
    iteration  = 0

    while True:
        iteration += 1
        log.info(f"\n{'='*50}")
        log.info(f"Iteration {iteration} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        try:
            pmids = search_pubmed(query, max_results)

            # Process in batches
            for i in range(0, len(pmids), PUBMED_BATCH_SIZE):
                batch = pmids[i: i + PUBMED_BATCH_SIZE]
                log.info(f"Fetching batch {i//PUBMED_BATCH_SIZE + 1} ({len(batch)} articles)…")
                articles = fetch_pubmed_articles(batch)
                sent = publish_articles(producer, articles)
                total_sent += sent
                log.info(f"Batch done: {sent}/{len(batch)} sent | Total: {total_sent}")
                time.sleep(1)   # NCBI rate limit: max 3 req/s without API key

        except requests.RequestException as e:
            log.error(f"API error: {e}")
        except Exception as e:
            log.exception(f"Unexpected error: {e}")

        producer.flush()
        log.info(f"\nTotal published so far: {total_sent} articles")

        if not loop:
            break
        log.info(f"Waiting {interval}s before next batch…")
        time.sleep(interval)

    producer.close()
    log.info(f"Producer stopped. Total sent: {total_sent}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PubMed → Kafka producer")
    p.add_argument("--query",    default=PUBMED_DEFAULT_QUERY, help="PubMed search query")
    p.add_argument("--max",      type=int, default=100,        help="Max articles to fetch")
    p.add_argument("--interval", type=int, default=60,         help="Seconds between loops")
    p.add_argument("--loop",     action="store_true",          help="Run continuously")
    args = p.parse_args()

    run(query=args.query, max_results=args.max, interval=args.interval, loop=args.loop)
