# Early Warning System for Predictive Cyber Threat Detection

**MSc Project — University of Lagos (Unilag)**  
**Author:** Oluwasegun Adeniyi  
**Supervisor:** Prof. N.A. Azeez  
**Department:** Cybersecurity and Software Engineering

---

## Overview

This project builds an early warning system that automatically detects and classifies emerging cyber threats by mining cybersecurity discussions from Stack Overflow. It uses Latent Dirichlet Allocation (LDA) for unsupervised threat topic discovery and XGBoost for classification, achieving **75.65% accuracy** across 10 threat categories on 92,771 real posts.

---

## Pipeline

| Script            | What it does                                                          |
| ----------------- | --------------------------------------------------------------------- |
| `collector.py`    | Pulls cybersecurity posts from Stack Overflow API (2015–2026)         |
| `preprocessor.py` | Cleans text: strips HTML, tokenises, lemmatises, builds Gensim corpus |
| `topic_model.py`  | Runs LDA topic modelling, selects optimal K via coherence sweep       |
| `classifier.py`   | Trains Random Forest + XGBoost, generates evaluation figures          |

---

## Results

| Model         | Accuracy   | Precision (macro) | Recall (macro) | F1 (macro) | Weighted F1 |
| ------------- | ---------- | ----------------- | -------------- | ---------- | ----------- |
| Random Forest | 57.88%     | 53.82%            | 65.64%         | 55.03%     | 56.02%      |
| **XGBoost**   | **75.65%** | **74.85%**        | **69.13%**     | **71.28%** | **75.32%**  |

- **Dataset:** 92,771 unique Stack Overflow posts (10 security tags)
- **Topics discovered:** 10 (K selected by C_v coherence = 0.5348)
- **Date range:** January 2015 – October 2026

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
