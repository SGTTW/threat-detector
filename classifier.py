"""
classifier.py  —  Threat Classification (10 Classifiers × 10 Metrics)
MSc Project: Early Warning System for Predictive Cyber Threat Detection

Usage:
    python classifier.py

Input:  data/posts_clean.csv, data/topic_assignments.csv
Output: figures/fig_confusion_rf.png, figures/fig_confusion_xgb.png,
        figures/fig_roc_curve.png, figures/fig_feature_importance.png,
        classification_report.txt, results_table.csv
"""

import csv
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import lightgbm as lgb


os.makedirs("figures", exist_ok=True)


def compute_metrics(clf_name, y_true, y_pred, y_proba, classes, oob=None):
    """Compute all 10 evaluation metrics for one classifier."""
    p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)

    # AUC-ROC (macro OVR)
    y_bin = label_binarize(y_true, classes=classes)
    try:
        auc_roc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_roc = float("nan")

    # Log loss
    try:
        ll = log_loss(y_true, y_proba)
    except Exception:
        ll = float("nan")

    # Brier score (mean over classes)
    try:
        bs = np.mean([
            brier_score_loss(y_bin[:, i], y_proba[:, i])
            for i in range(len(classes))
        ])
    except Exception:
        bs = float("nan")

    return {
        "Classifier": clf_name,
        "F1":         round(f1,  4),
        "AUC-ROC":    round(auc_roc, 4),
        "Accuracy":   round(acc, 4),
        "Precision":  round(p,   4),
        "Recall":     round(r,   4),
        "MCC":        round(mcc, 4),
        "Kappa":      round(kap, 4),
        "LogLoss":    round(ll,  4),
        "Brier":      round(bs,  4),
        "OOB":        round(oob, 4) if oob is not None else "N/A",
    }


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

    ids = [qid for qid in posts if qid in topics]
    ids.sort(key=lambda x: int(posts[x]["creation_date"] or 0))

    texts  = [posts[qid]["tokens"] for qid in ids]
    labels = [topics[qid] for qid in ids]

    print(f"  {len(ids):,} posts, {len(set(labels))} classes")
    print(f"  Class distribution: {dict(sorted(Counter(labels).items()))}")

    # ── Chronological 70 / 15 / 15 split ─────────────────────────────────────
    n  = len(ids)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    X_train_raw, y_train = texts[:t1],  labels[:t1]
    X_val_raw,   y_val   = texts[t1:t2], labels[t1:t2]
    X_test_raw,  y_test  = texts[t2:],   labels[t2:]
    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ── TF-IDF features ───────────────────────────────────────────────────────
    print("Fitting TF-IDF ...")
    tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train_raw)
    X_val   = tfidf.transform(X_val_raw)
    X_test  = tfidf.transform(X_test_raw)

    classes = sorted(set(labels))
    cw      = compute_class_weight("balanced", classes=np.array(classes), y=y_train)
    cwd     = dict(zip(classes, cw))

    # ── Define all 10 classifiers ─────────────────────────────────────────────
    # SVM wrapped for probability estimates (required for AUC / log-loss)
    svm_base = LinearSVC(max_iter=2000, random_state=42, class_weight=cwd)
    svm_cal  = CalibratedClassifierCV(svm_base, cv=3)

    classifiers = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, 
            class_weight=cwd, random_state=42, n_jobs=-1)),
        ("Multinomial NB",      MultinomialNB(alpha=1.0)),
        ("SVM",                 svm_cal),
        ("KNN",                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ("Decision Tree",       DecisionTreeClassifier(
            max_depth=20, class_weight=cwd, random_state=42)),
        ("Random Forest",       RandomForestClassifier(
            n_estimators=200, max_depth=30,
            class_weight=cwd, random_state=42, n_jobs=-1, oob_score=True)),
     ("Extra Trees", ExtraTreesClassifier(
    n_estimators=100, max_depth=30,
    class_weight=cwd, random_state=42, n_jobs=-1,
    bootstrap=True, oob_score=True)),   # bootstrap=True required for oob_score
        ("Gradient Boosting",   GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)),
        ("XGBoost",             xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42, n_jobs=-1)),
        ("LightGBM",            lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05,
            num_leaves=63, random_state=42, n_jobs=-1,
            class_weight=cwd, verbose=-1)),
    ]

    # ── Train, predict, evaluate each ─────────────────────────────────────────
    all_results = []
    rf_model = xgb_model = None   # keep for figures

    for name, clf in classifiers:
        print(f"\nTraining {name} ...")

        # XGBoost uses eval_set for early-stopping check
        if name == "XGBoost":
            clf.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)], verbose=False)
        else:
            clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        oob = None
        if hasattr(clf, "oob_score_"):
            oob = clf.oob_score_

        metrics = compute_metrics(name, y_test, preds, proba, classes, oob)
        all_results.append({**metrics, "_preds": preds, "_proba": proba})

        print(f"  F1={metrics['F1']:.4f}  AUC={metrics['AUC-ROC']:.4f}  Acc={metrics['Accuracy']:.4f}")
        print(classification_report(y_test, preds, digits=4))

        if name == "Random Forest":
            rf_model = (clf, preds, proba)
        if name == "XGBoost":
            xgb_model = (clf, preds, proba)

    # ── Save full text report ─────────────────────────────────────────────────
    with open("classification_report.txt", "w") as f:
        for r in all_results:
            f.write(f"=== {r['Classifier']} ===\n")
            f.write(classification_report(y_test, r["_preds"], digits=4))
            f.write("\n")
    print("\nSaved: classification_report.txt")

    # ── Save results CSV (for updating tables) ────────────────────────────────
    metric_cols = ["Classifier", "F1", "AUC-ROC", "Accuracy",
                   "Precision", "Recall", "MCC", "Kappa", "LogLoss", "Brier", "OOB"]
    with open("results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print("Saved: results_table.csv")

    # ── Figure: Confusion matrices (RF + XGBoost) ─────────────────────────────
    topic_labels = [f"T{i+1}" for i in range(len(classes))]

    for label, (clf, preds, _), fname in [
        ("Random Forest", rf_model,  "figures/fig_confusion_rf.png"),
        ("XGBoost",       xgb_model, "figures/fig_confusion_xgb.png"),
    ]:
        cm = confusion_matrix(y_test, preds, labels=classes)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(topic_labels, fontsize=9)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(topic_labels, fontsize=9)
        ax.set_xlabel("Predicted Topic", fontsize=11)
        ax.set_ylabel("True Topic", fontsize=11)
        ax.set_title(f"Confusion Matrix — {label}", fontsize=13, fontweight="bold")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > cm.max()/2 else "black")
        fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close()
        print(f"Saved: {fname}")

    # ── Figure: ROC curves per topic (XGBoost) ───────────────────────────────
    _, _, xgb_proba = xgb_model
    y_test_bin = label_binarize(y_test, classes=classes)
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], xgb_proba[:, i])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % 10], linewidth=1.5,
                label=f"T{c+1} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves per Threat Topic — XGBoost", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig_roc_curve.png", dpi=150); plt.close()
    print("Saved: fig_roc_curve.png")

    # ── Figure: Feature importance — XGBoost top 20 ──────────────────────────
    xgb_clf = xgb_model[0]
    feat_names   = tfidf.get_feature_names_out()
    importances  = xgb_clf.feature_importances_
    top_idx      = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feat_names[i] for i in top_idx],
            [importances[i] for i in top_idx], color="#2563EB")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Top 20 Most Important Features — XGBoost", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig("figures/fig_feature_importance.png", dpi=150); plt.close()
    print("Saved: fig_feature_importance.png")

    # ── Summary table in terminal ─────────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'Classifier':<22} {'F1':>7} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'MCC':>7} {'Kappa':>7} {'LogL':>8} {'Brier':>7} {'OOB':>7}")
    print("="*90)
    for r in all_results:
        print(f"{r['Classifier']:<22} {r['F1']:>7} {r['AUC-ROC']:>7} {r['Accuracy']:>7} "
              f"{r['Precision']:>7} {r['Recall']:>7} {r['MCC']:>7} {r['Kappa']:>7} "
              f"{r['LogLoss']:>8} {r['Brier']:>7} {str(r['OOB']):>7}")
    print("="*90)
    print("\nDone. Check results_table.csv for the full table to paste into Chapter 4.")


if __name__ == "__main__":
    main()