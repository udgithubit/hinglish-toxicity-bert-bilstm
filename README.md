# Hinglish Toxicity Detection: BERT + BiLSTM

Why a plain BERT classifier misses toxic comments in code-mixed
Hindi-English (Hinglish) text — and how adding a bidirectional LSTM
head fixes it.

## The problem

Most toxicity classifiers are trained and tuned on English. Hindi
social media text is frequently code-mixed (Hinglish), has free word
order, and mixes scripts (Devanagari + Latin). A flat BERT classifier
often misses toxic terms that shift position in the sentence — a
BiLSTM head, fed BERT's contextual embeddings, models that sequential
context and closes the gap.

## What's in this repo

- `bert_baseline.py` — BERT-only classifier (the baseline that misses
  certain code-mixed examples)
- `bert_bilstm.py` — BERT embeddings + BiLSTM head (the fix)
- `notebook.ipynb` — runnable Colab version, free-tier friendly
- `requirements.txt`

## Dataset

Uses a small public Hinglish dataset (HASOC / TRAC) — no proprietary
data required. See `data/` for the loading script.

## Results

Trained and evaluated on this repo's dataset (numbers here will
differ from the original research paper, which used a larger private
corpus — see [paper link] for those results).

## Background

This code is adapted from research on hybrid transformer architectures
for toxic Hindi content detection, published at ICMLDE 2025.
