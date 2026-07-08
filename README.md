# Hinglish Toxicity Detection: BERT + BiLSTM

Why a plain BERT classifier and a BERT+BiLSTM hybrid perform the way they
do on code-mixed Hindi-English (Hinglish) toxic content — and what that
reveals about when architectural complexity actually helps.

## Related writeups

- [Full analysis: BERT vs BERT+BiLSTM — an honest result](https://dev.to/udgithubit/bert-vs-bertbilstm-an-honest-result-on-hinglish-toxicity-detection-1c1m/stats)
- [Step-by-step tutorial: Fine-tuning mBERT for Hinglish toxicity classification](https://dev.to/udgithubit/how-to-fine-tune-mbert-for-hinglish-toxicity-classification-a-step-by-step-guide-f27)

## The problem

Most toxicity classifiers are trained and tuned on English. Hindi social
media text is frequently code-mixed (Hinglish), has free word order, and
mixes scripts (Devanagari + Latin). This repo explores whether adding a
bidirectional LSTM head on top of BERT's contextual embeddings improves
detection — and reports an honest result, not just a clean win.

## What's in this repo

- `bert_baseline.py` — plain BERT classifier (pooled output → linear layer)
- `bert_bilstm.py` — two hybrid variants:
  - **BERT → BiLSTM**: full token-level BERT embeddings fed through a BiLSTM
  - **BiLSTM → BERT (frozen)**: BERT frozen, only the single pooled vector
    fed through a BiLSTM
- `hinglish-tutorial.md` — step-by-step walkthrough of training the
  baseline BERT classifier from scratch, written as a companion to the
  analysis above
- `data/hinglish_toxicity_sample_500.csv` — 500-row balanced sample (250
  toxic / 250 non-toxic), filtered from a larger 60k-post corpus for label
  quality (see **Dataset** below)
- `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
python bert_baseline.py
python bert_bilstm.py
```

Runs on CPU; each script takes roughly 20 minutes on a typical laptop.

## Dataset

This sample is drawn from a larger 60,000-post Hindi social media dataset
(30k Reddit + 30k X/Twitter posts), used in a separate research paper on
Hindi toxic content detection ("Hybrid Transformer-Based Neural
Architectures for Toxic Hindi Social Media," accepted and presented at
CSCT 2025). Manual inspection showed meaningful label noise in the full
corpus — a portion of "toxic"-labeled rows were neutral news or opinion
text, not actually abusive language. To keep this tutorial's results
interpretable, this 500-row sample was **keyword-filtered and length-capped**
to select clearly-labeled examples of each class, then manually
spot-checked. This makes the classification task easier than the full,
noisy corpus — a deliberate tradeoff for a clear teaching example, not a
benchmark claim.

## Results

| Model                          | Accuracy | Precision | Recall | F1   |
|---------------------------------|----------|-----------|--------|------|
| Plain BERT (baseline)           | 0.96     | 0.98      | 0.94   | 0.96 |
| BERT → BiLSTM                   | 0.95     | 1.00      | 0.88   | 0.94 |
| BiLSTM → BERT (frozen)          | 0.65     | 0.56      | 0.93   | 0.70 |

**Notable finding:** on this clean, filtered sample, the plain BERT
baseline matches or slightly edges out the BERT→BiLSTM hybrid. This
doesn't contradict prior published results — it suggests the BiLSTM's
sequential modeling advantage matters more on **larger, noisier, more
ambiguous** real-world data than on a small, clearly-labeled sample. The
frozen BiLSTM→BERT variant performs far worse, which makes sense
architecturally: freezing BERT and passing only a single pooled vector
through a bidirectional LSTM leaves no real sequence to model.

**Note on the full-scale result:** this repo is a small, clean teaching
demo, not a benchmark. On the complete 60,000-record corpus (the messier,
real-world version of this data), the BiLSTM head does show a measurable
improvement — reaching 94.2% accuracy in a separate research paper
("Hybrid Transformer-Based Neural Architectures for Toxic Hindi Social
Media," accepted and presented at CSCT 2025). That's consistent with the
explanation above: a BiLSTM's sequential modeling earns its keep on
ambiguous, noisy data, and this repo's 500-row sample was deliberately
filtered to remove exactly that ambiguity.

## License

MIT
