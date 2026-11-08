# Early Warning System for Predictive Cyber Threat Detection

**MSc Project — University of Lagos (Unilag)**  
**Author:** Oluwasegun Adeniyi  
**Supervisor:** Prof. N.A. Azeez  
**Department:** Cybersecurity and Software Engineering

---

## Overview

This project builds an early warning system that automatically detects and classifies emerging cyber threats by mining cybersecurity discussions from Stack Overflow. It uses Latent Dirichlet Allocation (LDA) for unsupervised threat topic discovery and Support Vector Machine (SVM) for classification, achieving **83.48% accuracy** across 10 threat categories on 92,771 real posts.

---

## Pipeline

| Script            | What it does                                                          |
| ----------------- | --------------------------------------------------------------------- |
| `collector.py`    | Pulls cybersecurity posts from Stack Overflow API (2015–2026)         |
| `preprocessor.py` | Cleans text: strips HTML, tokenises, lemmatises, builds Gensim corpus |
| `topic_model.py`  | Runs LDA topic modelling, selects optimal K via coherence sweep       |
| `classifier.py`   | Trains 10 classifiers, evaluates across 10 metrics, generates figures |

---

## Results

| Model               | Accuracy   | Precision (macro) | Recall (macro) | F1 (macro) | Weighted F1 |
| ------------------- | ---------- | ----------------- | -------------- | ---------- | ----------- |
| Logistic Regression | 82.88%     | 76.87%            | 84.61%         | 79.83%     | 83.08%      |
| Multinomial NB      | 76.11%     | 80.11%            | 63.81%         | 66.29%     | 74.82%      |
| **SVM**             | **83.48%** | **83.11%**        | **77.75%**     | **79.94%** | **83.30%**  |
| KNN                 | 61.83%     | 57.07%            | 63.10%         | 59.23%     | 61.74%      |
| Decision Tree       | 47.15%     | 43.40%            | 48.31%         | 44.52%     | 47.30%      |
| Random Forest       | 57.88%     | 53.82%            | 65.64%         | 55.03%     | 56.02%      |
| Extra Trees         | 57.95%     | 54.36%            | 67.59%         | 55.27%     | 56.10%      |
| Gradient Boosting   | 67.07%     | 67.32%            | 59.29%         | 61.98%     | 66.42%      |
| XGBoost             | 70.42%     | 70.84%            | 63.59%         | 66.10%     | 69.89%      |
| LightGBM            | 79.77%     | 76.52%            | 77.00%         | 76.60%     | 79.74%      |

- **Best model:** SVM (F1 = 0.7994, Accuracy = 83.48%)
- **Dataset:** 92,771 unique Stack Overflow posts (10 security tags)
- **Topics discovered:** 10 (K selected by C_v coherence = 0.5348)
- **Date range:** January 2015 – June 2026

---

## Setup

### Requirements

- Python 3.10+
- Git

### Install dependencies

```bash
pip install -r requirements.txt
```

### Get a Stack Overflow API key

1. Go to [stackapps.com](https://stackapps.com)
2. Register an app to get a free API key
3. Copy `.env.example` to `.env` and put your key in it:

### Run the full pipeline

```bash
python collector.py       # ~30 mins, collects 90K+ posts
python preprocessor.py    # ~15 mins, cleans and tokenises
python topic_model.py     # ~40 mins, discovers threat topics
python classifier.py      # ~15 mins, trains and evaluates models
```

---

## Output Files

| File                                 | Description                                   |
| ------------------------------------ | --------------------------------------------- |
| `data/posts_raw.jsonl`               | Raw collected posts (JSON Lines)              |
| `data/posts_clean.csv`               | Cleaned tokens + metadata                     |
| `data/topic_assignments.csv`         | Each post labelled with dominant threat topic |
| `figures/fig_coherence.png`          | LDA coherence curve (K selection)             |
| `figures/fig_topic_words.png`        | Top keywords per threat topic                 |
| `figures/fig_topic_trends.png`       | Threat topic volume trends 2015–2026          |
| `figures/fig_confusion_rf.png`       | Random Forest confusion matrix                |
| `figures/fig_confusion_xgb.png`      | XGBoost confusion matrix                      |
| `figures/fig_roc_curve.png`          | ROC curves per threat topic                   |
| `figures/fig_feature_importance.png` | Top 20 most important features                |
| `classification_report.txt`          | Full precision/recall/F1 report               |

---

## Threat Topics Discovered

| Topic | Key Terms                                 | Category              |
| ----- | ----------------------------------------- | --------------------- |
| 1     | server, run, test, window                 | Server/Infrastructure |
| 2     | key, private, hash, encrypted             | Cryptography          |
| 3     | password, database, username, login       | Credential Attacks    |
| 4     | malware, script, php, malicious           | Malware & Injection   |
| 5     | auth, session, cookie, laravel            | Authentication        |
| 6     | authentication, authorization, azure, web | Cloud Authorization   |
| 7     | login, url, redirect, form, browser       | XSS & Phishing        |
| 8     | certificate, apache, http, ssl            | SSL/TLS Security      |
| 9     | firebase, android, aws, device            | Mobile & Cloud        |
| 10    | token, api, authorization, header         | API Security          |