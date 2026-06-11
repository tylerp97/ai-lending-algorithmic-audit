# AI Lending Algorithmic Audit

An end-to-end audit of a credit-risk classification model, examining whether a logistic regression trained on the [Statlog German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) produces fair outcomes across demographic subgroups.

## Motivation

Lending models are among the highest-stakes applications of machine learning — approval or denial directly affects financial access. This project investigates whether a simple, interpretable model can achieve acceptable predictive performance while remaining fair across protected attributes (gender, age, and foreign worker status).

## Data

- **Source:** UCI Machine Learning Repository — Statlog (German Credit Data)
- **Size:** 1,000 samples, 20 features, 1 binary target (`good` / `bad` credit risk)
- **Access:** Loaded programmatically via the `ucimlrepo` package — no local data file required

## Methods

1. **Preprocessing** — feature renaming, one-hot encoding of 13 categorical variables (48 encoded features total)
2. **Modeling** — logistic regression with a 50/50 stratified train/test split
3. **Fairness Analysis**
   - Approval rate disparities across subgroups (gender, age bucket, foreign worker status)
   - Confusion matrix breakdown per protected group
   - Formal fairness metrics via IBM AIF360: Statistical Parity Difference, Disparate Impact, Equal Opportunity Difference, Average Odds Difference

## Key Findings

- Baseline accuracy: ~75.4%
- The model exhibits measurable disparate impact across age and gender subgroups
- Simple logistic regression trades off predictive performance against fairness in ways that require active mitigation

## Project Structure

```
ai-lending-algorithmic-audit/
├── README.md
├── requirements.txt
├── notebooks/
│   └── ai_lending_algorithmic_audit.ipynb   # Full analysis
├── data/
│   ├── raw/          # Original source data (loaded at runtime via ucimlrepo)
│   └── processed/    # Cleaned and transformed data
├── src/              # Reusable Python modules
└── reports/
    └── figures/      # Exported charts and visualizations
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ai_lending_algorithmic_audit.ipynb
```

## Tools

Python, Pandas, NumPy, scikit-learn, IBM AIF360, Matplotlib, ucimlrepo
