# Comparative Analysis of Neural Machine Translation Architectures (German -> English)

A comparative study of four RNN-based NMT architectures, trained and evaluated on identical
German–English data and hyperparameters:

1. **Plain RNN** — a single, shared-weight recurrent network (no separate encoder/decoder networks, no attention).
2. **Encoder-Decoder** — the classical Seq2Seq architecture with separate Encoder and Decoder RNNs.
3. **Encoder-Decoder + Additive Attention** — Bahdanau-style attention (`v^T tanh(W1·h + W2·s)`).
4. **Encoder-Decoder + Multiplicative Attention** — Luong-style attention (`s^T · Wa · h`).

`random_state = 20` is used throughout for reproducibility (dataset subsampling, train/val/test
split, DataLoader shuffling, and weight initialization seeding).

---

## Repository structure

```
nmt-comparison/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── deu.txt                 # ManyThings/Tatoeba German-English pairs (gitignored — see below)
│   └── _about.txt               # dataset attribution, as shipped by ManyThings.org
├── src/
│   ├── data_prep.py              # data loading, cleaning, PAD/SOS/EOS/UNK vocab, 80/10/10 split (random_state=20)
│   └── train_utils.py            # training loop, evaluation, BLEU, perplexity, inference latency plotting, results bookkeeping
├── notebooks/
│   ├── 01_RNN.ipynb
│   ├── 02_Encoder_Decoder.ipynb
│   ├── 03_Additive_Attention.ipynb
│   ├── 04_Multiplicative_Attention.ipynb
│   └── 05_Comparison.ipynb      # loads outputs/results.json, combined comparison + discussion/conclusion
├── outputs/
│   ├── figures/                 # loss curves + attention heatmaps (PNG)
│   └── results.json             # metrics for all 4 models, written by notebooks 01-04, read by 05
└── report/
    └── NMT_Comparative_Report.docx
```

## Dataset

**Source:** [ManyThings.org / Tatoeba Project](https://www.manythings.org/anki/) - `deu-eng.zip`
(German–English sentence pairs). Each line is `English<TAB>German<TAB>attribution`.

Download it yourself with:

```bash
wget https://www.manythings.org/anki/deu-eng.zip -O data/deu-eng.zip
unzip data/deu-eng.zip -d data/
rm data/deu-eng.zip
```

This produces `data/deu.txt` (~331k sentence pairs). The data file is **gitignored** - download it
fresh rather than committing ~50MB of corpus text to the repo.

### Why a subsample, not the full 331k pairs?

All four models were trained on the same 8,000-pair subset using random_state=20, with sentences limited to 10 tokens after preprocessing. We used a smaller subset because training the full filtered dataset of around 272k pairs on a single CPU would take too long for this project. Using the same data split and hyperparameters for all four models keeps the comparison fair, although the smaller dataset limits the overall translation quality.

If feasible, you can increase or remove SUBSET_SIZE in src/data_prep.py to train on more data. This should improve the BLEU scores for the models.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"   # only needed if you extend the BLEU/tokenization code
```

## Reproducing the experiments

Run the notebooks in order - each is self-contained (Title/Objective/Theory/Code/Discussion,
matching the course's lab-report format) but they share `src/data_prep.py` and `src/train_utils.py` so preprocessing and
splitting are guaranteed identical:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_RNN.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Encoder_Decoder.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_Additive_Attention.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_Multiplicative_Attention.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_Comparison.ipynb
```

Or open them in Jupyter/VS Code/Colab and run cells top-to-bottom. Each of notebooks 01–04 appends
its metrics (loss curves, parameter count, BLEU, etc.) to `outputs/results.json`; notebook 05 reads
that file to build the final comparison table, plots, and combined Discussion/Conclusion - so 01–04
must be run at least once before 05.

**Note on runtime:** on a single CPU core, each of notebooks 01–04 takes roughly 3-4 minutes for
40 epochs on the 8,000-pair subset. A GPU will finish each in well under a minute.

## Hyperparameters (identical across all 4 models)

| Setting | Value |
|---|---|
| `random_state` | 20 |
| Subset size | 8,000 pairs |
| Train / Val / Test split | 6,400 / 800 / 800 |
| `MAX_LENGTH` | 10 tokens |
| `hidden_size` | 128 |
| Special tokens | `PAD=0`, `SOS=1`, `EOS=2`, `UNK=3` (padding excluded from loss via `ignore_index`; unseen words at eval time map to `UNK` instead of raising an error) |
| `batch_size` | 64 |
| Optimizer | Adam, lr = 0.001 |
| Loss | NLLLoss (with teacher forcing) |
| Epochs | 40 |

## Evaluation

- **Quantitative (accuracy):** training/validation NLL loss curves, corpus BLEU-4 (with smoothing) on a held-out test set, and perplexity (`exp(loss)`) for train/validation, reported both at the final epoch and at each model's best epoch.
- **Quantitative (efficiency):** average inference latency (ms/sentence) and throughput (sentences/sec) for greedy decoding, measured over 100 test-set sentences after a short warmup — lets the report discuss accuracy-vs-speed tradeoffs (e.g. whether attention's BLEU gains are "worth" its extra per-step compute), not just raw accuracy.
- **Qualitative:** 10 randomly sampled test-set translations per model.
- **Interpretability:** attention-weight heatmaps for the two attention-based models (03, 04).

Combined comparison plots for all four models — BLEU/parameter count, training/validation loss
curves, validation perplexity, and BLEU-vs-latency — are generated in `05_Comparison.ipynb` and
saved to `outputs/figures/`.

## Report

The full research-article-style report including Abstract, Introduction, Methodology, Experimental Setup,
Results, per-model Discussion, and a combined comparative Discussion & Conclusion — is at
`report/NMT_Comparative_Report.docx`.

## Attribution

German–English sentence pairs are © their contributors on [Tatoeba.org](https://tatoeba.org),
distributed under CC-BY 2.0 (France) via [ManyThings.org](https://www.manythings.org/anki/). See
`data/_about.txt` for the exact terms.
