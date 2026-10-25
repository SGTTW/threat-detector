"""
topic_model.py  —  LDA Topic Modelling
MSc Project: Early Warning System for Predictive Cyber Threat Detection

Usage:
    python topic_model.py

Input:  dictionary.gensim, corpus.mm, posts_clean.csv
Output: lda_model/           — saved LDA model
        fig_coherence.png    — coherence curve (K selection)
        fig_topic_words.png  — top words per topic
        fig_topic_trends.png — topic volume over time
        topic_assignments.csv — each post's dominant topic
"""

import csv
import os
import warnings
warnings.filterwarnings("ignore")

import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from collections import defaultdict

DICT_FILE    = "models/dictionary.gensim"
CORPUS_FILE  = "models/corpus.mm"
CLEAN_CSV    = "data/posts_clean.csv"
MODEL_DIR    = "models/lda_model"
K_MIN, K_MAX = 8, 20
PASSES       = 10
RANDOM_STATE = 42


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dictionary and corpus ...")
    dictionary = corpora.Dictionary.load(DICT_FILE)
    corpus     = list(corpora.MmCorpus(CORPUS_FILE))   # load into memory once

    print("Loading cleaned posts ...")
    posts = []
    with open(CLEAN_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            posts.append(row)
    print(f"  {len(posts):,} posts loaded")

    tokenised = [row["tokens"].split() for row in posts]

    # ── Coherence sweep ───────────────────────────────────────────────────────
    print(f"\nRunning coherence sweep K={K_MIN}..{K_MAX} ...")
    k_values, coherences = [], []

    for k in range(K_MIN, K_MAX + 1):
        print(f"  K={k} ...", end=" ", flush=True)
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=PASSES,
            random_state=RANDOM_STATE,
            alpha="auto",
            eta="auto",
        )
        cm = CoherenceModel(
            model=lda, texts=tokenised,
            dictionary=dictionary, coherence="c_v",
            processes=1,        # single process — avoids Windows spawn issue
        )
        c = cm.get_coherence()
        k_values.append(k)
        coherences.append(c)
        print(f"C_v={c:.4f}")

    best_k = k_values[coherences.index(max(coherences))]
    print(f"\nBest K = {best_k}  (C_v = {max(coherences):.4f})")

    # ── Figure 1: Coherence curve ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_values, coherences, marker="o", linewidth=2, color="#2563EB")
    ax.axvline(best_k, color="#DC2626", linestyle="--", linewidth=1.5,
               label=f"Best K={best_k}")
    ax.set_xlabel("Number of Topics (K)", fontsize=12)
    ax.set_ylabel("C_v Coherence Score", fontsize=12)
    ax.set_title("LDA Topic Coherence vs Number of Topics", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_coherence.png", dpi=150)
    plt.close()
    print("Saved: fig_coherence.png")

    # ── Train final model at best K ───────────────────────────────────────────
    print(f"\nTraining final LDA model at K={best_k} ...")
    lda_final = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        passes=PASSES,
        random_state=RANDOM_STATE,
        alpha="auto",
        eta="auto",
    )
    lda_final.save(os.path.join(MODEL_DIR, "lda_final"))
    print(f"Saved: {MODEL_DIR}/lda_final")

    # ── Figure 2: Topic word bars ─────────────────────────────────────────────
    print("Generating topic-word chart ...")
    colors = list(mcolors.TABLEAU_COLORS.values())
    cols   = 2
    rows   = (best_k + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2.8))
    axes = axes.flatten()

    for t in range(best_k):
        top_words = lda_final.show_topic(t, topn=10)
        words  = [w for w, _ in top_words][::-1]
        scores = [s for _, s in top_words][::-1]
        ax = axes[t]
        ax.barh(words, scores, color=colors[t % len(colors)])
        ax.set_title(f"Topic {t+1}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for i in range(best_k, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Top 10 Keywords per Threat Topic (LDA)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/fig_topic_words.png", dpi=150)
    plt.close()
    print("Saved: fig_topic_words.png")

    # ── Assign dominant topic to each post ────────────────────────────────────
    print("Assigning topics to posts ...")
    assignments = []
    for bow in corpus:
        dist = lda_final.get_document_topics(bow, minimum_probability=0)
        dominant = max(dist, key=lambda x: x[1])[0]
        assignments.append(dominant)

    with open("data/topic_assignments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "creation_date", "dominant_topic"])
        for i, post in enumerate(posts):
            writer.writerow([post["question_id"], post["creation_date"], assignments[i]])
    print("Saved: topic_assignments.csv")

    # ── Figure 3: Topic trends over time ─────────────────────────────────────
    print("Generating topic trend chart ...")
    year_topic = defaultdict(lambda: defaultdict(int))
    for i, post in enumerate(posts):
        ts   = int(post["creation_date"]) if post["creation_date"] else 0
        year = datetime.utcfromtimestamp(ts).year if ts else 0
        if 2015 <= year <= 2025:
            year_topic[year][assignments[i]] += 1

    years = sorted(year_topic.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    for t in range(best_k):
        counts = [year_topic[y][t] for y in years]
        ax.plot(years, counts, marker="o", linewidth=1.8,
                label=f"Topic {t+1}", color=colors[t % len(colors)])

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_title("Cyber Threat Topic Volume Trends (2015–2025)", fontsize=13, fontweight="bold")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_topic_trends.png", dpi=150)
    plt.close()
    print("Saved: fig_topic_trends.png")

    print("\n── Topic word summaries ─────────────────────────────────────────")
    for t in range(best_k):
        top = ", ".join([w for w, _ in lda_final.show_topic(t, topn=8)])
        print(f"  Topic {t+1:2d}: {top}")

    print("\nDone. Run classifier.py next.")


if __name__ == "__main__":
    main()