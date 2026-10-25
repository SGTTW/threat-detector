"""
classifier.py  —  Threat Classification (Random Forest + XGBoost)
MSc Project: Early Warning System for Predictive Cyber Threat Detection

Usage:
    python classifier.py

Input:  posts_clean.csv, topic_assignments.csv
Output: fig_confusion_rf.png, fig_confusion_xgb.png,
        fig_roc_curve.png, fig_feature_importance.png,
        classification_report.txt
"""

import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data ...")
    posts, topics = {}, {}

    with open("data/posts_clean.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            posts[row["question_id"]] = row

    with open("data/topic_assignments.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            topics[row["question_id"]] = int(row["dominant_topic"])

    # Align
    ids = [qid for qid in posts if qid in topics]
    ids.sort(key=lambda x: int(posts[x]["creation_date"] or 0))

    texts  = [posts[qid]["tokens"] for qid in ids]
    labels = [topics[qid] for qid in ids]
    dates  = [int(posts[qid]["creation_date"] or 0) for qid in ids]

    print(f"  {len(ids):,} posts, {len(set(labels))} classes")
    print(f"  Class distribution: {dict(sorted(Counter(labels).items()))}")

    # ── Chronological 70/15/15 split ─────────────────────────────────────────
    n = len(ids)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    X_train_raw, y_train = texts[:t1], labels[:t1]
    X_val_raw,   y_val   = texts[t1:t2], labels[t1:t2]
    X_test_raw,  y_test  = texts[t2:],  labels[t2:]
    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ── TF-IDF features ───────────────────────────────────────────────────────
    print("Fitting TF-IDF ...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train_raw)
    X_val   = tfidf.transform(X_val_raw)
    X_test  = tfidf.transform(X_test_raw)

    classes = sorted(set(labels))
    cw = compute_class_weight("balanced", classes=np.array(classes), y=y_train)
    class_weight_dict = dict(zip(classes, cw))

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=30,
        class_weight=class_weight_dict,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    print("  RF done.")
    print(classification_report(y_test, rf_preds, digits=4))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("Training XGBoost ...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    print("  XGBoost done.")
    print(classification_report(y_test, xgb_preds, digits=4))

    # ── Save classification report ────────────────────────────────────────────
    with open("classification_report.txt", "w") as f:
        f.write("=== Random Forest ===\n")
        f.write(classification_report(y_test, rf_preds, digits=4))
        f.write("\n=== XGBoost ===\n")
        f.write(classification_report(y_test, xgb_preds, digits=4))
    print("Saved: classification_report.txt")

    # ── Figure: Confusion matrices ────────────────────────────────────────────
    topic_labels = [f"T{i+1}" for i in range(len(classes))]

    for name, preds, fname in [
        ("Random Forest", rf_preds, "figures/fig_confusion_rf.png"),
        ("XGBoost",       xgb_preds, "figures/fig_confusion_xgb.png"),
    ]:
        cm = confusion_matrix(y_test, preds, labels=classes)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(topic_labels, fontsize=9)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(topic_labels, fontsize=9)
        ax.set_xlabel("Predicted Topic", fontsize=11)
        ax.set_ylabel("True Topic", fontsize=11)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > cm.max()/2 else "black")
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")

    # ── Figure: ROC curve (macro avg, XGBoost) ────────────────────────────────
    y_test_bin = label_binarize(y_test, classes=classes)
    fpr_all, tpr_all, roc_auc_all = {}, {}, {}
    for i, c in enumerate(classes):
        fpr_all[i], tpr_all[i], _ = roc_curve(y_test_bin[:, i], xgb_proba[:, i])
        roc_auc_all[i] = auc(fpr_all[i], tpr_all[i])

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, c in enumerate(classes):
        ax.plot(fpr_all[i], tpr_all[i], color=colors[i % 10], linewidth=1.5,
                label=f"T{c+1} (AUC={roc_auc_all[i]:.3f})")
    ax.plot([0,1],[0,1],"k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves per Threat Topic — XGBoost", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_roc_curve.png", dpi=150)
    plt.close()
    print("Saved: fig_roc_curve.png")

    # ── Figure: Feature importance (XGBoost top 20) ───────────────────────────
    feat_names = tfidf.get_feature_names_out()
    importances = xgb_model.feature_importances_
    top_idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feat_names[i] for i in top_idx],
            [importances[i] for i in top_idx], color="#2563EB")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Top 20 Most Important Features — XGBoost", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/fig_feature_importance.png", dpi=150)
    plt.close()
    print("Saved: fig_feature_importance.png")

    print("\nDone. All figures and reports saved.")


if __name__ == "__main__":
    main()

    