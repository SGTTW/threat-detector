"""
preprocessor.py  —  Text Cleaning and Feature Preparation
MSc Project: Early Warning System for Predictive Cyber Threat Detection

Usage:
    python preprocessor.py

Input:  posts_raw.jsonl
Output: posts_clean.csv       — cleaned token strings + metadata
        dictionary.gensim     — Gensim token dictionary
        corpus.mm             — Gensim MmCorpus (bag-of-words)
"""

import json
import os
import re
import csv
import string
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import gensim.corpora as corpora

# ── NLTK data (download if missing) ─────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_JSONL    = "data/posts_raw.jsonl"
OUTPUT_CSV     = "data/posts_clean.csv"
DICT_FILE      = "models/dictionary.gensim"
CORPUS_FILE    = "models/corpus.mm"

MIN_TOKEN_FREQ = 5       # drop tokens appearing in fewer than this many posts
MAX_TOKEN_FRAC = 0.50    # drop tokens appearing in more than 50% of posts

STOP_EXTRA = {
    "anyone", "please", "thanks", "thank", "tried", "using", "getting",
    "want", "need", "know", "make", "get", "would", "could", "use",
    "used", "also", "even", "still", "like", "way", "thing", "something",
    "work", "working", "works", "look", "trying", "one", "two", "three",
    "seem", "seems", "example", "issue", "problem", "error", "help",
    "code", "file", "line", "function", "method", "class", "object",
    "return", "value", "variable", "type", "string", "number", "list",
}

lemmatizer = WordNetLemmatizer()
stop_en    = set(stopwords.words("english")) | STOP_EXTRA

# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean(title: str, body: str) -> list[str]:
    """Return a list of cleaned tokens from a post's title + body."""

    # 1. Strip HTML; remove <code> and <pre> blocks entirely
    soup = BeautifulSoup(body, "html.parser")
    for tag in soup.find_all(["code", "pre"]):
        tag.decompose()
    text = soup.get_text(separator=" ")

    # 2. Prepend title (it's the clearest signal)
    text = title + " " + text

    # 3. Lowercase
    text = text.lower()

    # 4. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 5. Remove standalone numbers but keep compound tokens (sha-256, cve-2023-1234)
    text = re.sub(r"\b\d+\b", " ", text)

    # 6. Remove punctuation except hyphens inside words
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)   # strip dangling hyphens

    # 7. Tokenise
    tokens = word_tokenize(text)

    # 8. Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in stop_en and len(t) > 2]

    # 9. Lemmatise
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


# ── Main ──────────────────────────────────────────────────────────────────────

def preprocess():
    print("Loading posts_raw.jsonl ...")
    posts = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            posts.append(json.loads(line))
    print(f"  {len(posts):,} posts loaded")

    print("Cleaning text ...")
    tokenised = []
    clean_rows = []
    for i, p in enumerate(posts):
        if i % 5000 == 0:
            print(f"  {i:,} / {len(posts):,}")
        tokens = clean(p.get("title", ""), p.get("body", ""))
        tokenised.append(tokens)
        clean_rows.append({
            "question_id":   p["question_id"],
            "creation_date": p["creation_date"],
            "score":         p["score"],
            "answer_count":  p["answer_count"],
            "has_accepted":  1 if p.get("accepted_answer_id") else 0,
            "tags":          "|".join(p.get("tags", [])),
            "tag_count":     len(p.get("tags", [])),
            "tokens":        " ".join(tokens),   # space-joined for easy reload
            "link":          p.get("link", ""),
        })

    # ── Build Gensim dictionary ───────────────────────────────────────────────
    print("Building dictionary ...")
    dictionary = corpora.Dictionary(tokenised)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=MIN_TOKEN_FREQ, no_above=MAX_TOKEN_FRAC)
    after = len(dictionary)
    print(f"  Vocabulary: {before:,} → {after:,} tokens after filtering")

    # ── Build corpus ──────────────────────────────────────────────────────────
    print("Building bag-of-words corpus ...")
    corpus = [dictionary.doc2bow(t) for t in tokenised]

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    dictionary.save(DICT_FILE)
    corpora.MmCorpus.serialize(CORPUS_FILE, corpus)
    print(f"Saved: {DICT_FILE}")
    print(f"Saved: {CORPUS_FILE}")

    # CSV with cleaned tokens + metadata
    fields = [
        "question_id", "creation_date", "score", "answer_count",
        "has_accepted", "tags", "tag_count", "tokens", "link"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_rows)
    print(f"Saved: {OUTPUT_CSV}")
    print("\nDone. Run topic_model.py next.")


if __name__ == "__main__":
    preprocess()