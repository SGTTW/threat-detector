"""
collector.py  —  Stack Overflow Security Post Collector
MSc Project: Early Warning System for Predictive Cyber Threat Detection

Usage:
    python collector.py

Output:
    posts_raw.jsonl   — one JSON object per line, one post per line
    posts.csv         — flat CSV of the fields used in analysis
"""

import json
import time
import csv
import os
from datetime import datetime
from stackapi import StackAPI

# ── Configuration ────────────────────────────────────────────────────────────

# Load KEY=VALUE pairs from .env (variables already set in the shell win)
if os.path.exists(".env"):
    with open(".env", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))

API_KEY   = os.environ.get("STACK_API_KEY")
SITE      = "stackoverflow"
PAGE_SIZE = 100
PAUSE     = 1.0

FROM_DATE = int(datetime(2015, 1, 1).timestamp())
TO_DATE   = int(datetime.now().timestamp())

TAGS = [
    "security",
    "web-security",
    "cryptography",
    "sql-injection",
    "xss",
    "authentication",
    "authorization",
    "vulnerability",
    "malware",
    "penetration-testing",
]

OUTPUT_JSONL = "data/posts_raw.jsonl"
OUTPUT_CSV   = "data/posts.csv"

# ── Collector ────────────────────────────────────────────────────────────────

def collect():
    if not API_KEY:
        raise SystemExit("Set the STACK_API_KEY environment variable to your Stack Apps key.")
    site = StackAPI(SITE, key=API_KEY)
    site.page_size = PAGE_SIZE
    site.max_pages = 10000

    all_posts = {}

    print(f"Starting collection for {len(TAGS)} tags...")
    print(f"Date range: 2015-01-01 → {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 60)

    for tag in TAGS:
        print(f"\n[{tag}] Fetching posts...")
        tag_count = 0

        try:
            results = site.fetch(
                "questions",
                fromdate=FROM_DATE,
                todate=TO_DATE,
                order="asc",
                sort="creation",
                tagged=tag,
                filter="withbody",
            )
        except Exception as e:
            print(f"  ERROR fetching [{tag}]: {e}")
            time.sleep(5)
            continue

        for post in results.get("items", []):
            pid = post.get("question_id")
            if pid and pid not in all_posts:
                all_posts[pid] = {
                    "question_id":        pid,
                    "title":              post.get("title", ""),
                    "body":               post.get("body", ""),
                    "tags":               post.get("tags", []),
                    "creation_date":      post.get("creation_date", 0),
                    "score":              post.get("score", 0),
                    "answer_count":       post.get("answer_count", 0),
                    "accepted_answer_id": post.get("accepted_answer_id", None),
                    "view_count":         post.get("view_count", 0),
                    "link":               post.get("link", ""),
                }
                tag_count += 1

        print(f"  → {tag_count} new posts (total so far: {len(all_posts)})")
        time.sleep(PAUSE)

    print(f"\nTotal unique posts collected: {len(all_posts)}")

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for post in all_posts.values():
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
    print(f"Saved: {OUTPUT_JSONL}")

    csv_fields = [
        "question_id", "title", "tags", "creation_date",
        "score", "answer_count", "accepted_answer_id", "view_count", "link"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for post in all_posts.values():
            row = dict(post)
            row["tags"] = "|".join(row.get("tags", []))
            writer.writerow(row)
    print(f"Saved: {OUTPUT_CSV}")
    print("\nDone. Run preprocessor.py next.")

if __name__ == "__main__":
    collect()