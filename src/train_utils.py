"""
train_utils.py
---------------
Training loop, evaluation, BLEU scoring, plotting, and results are
kept separate from data_prep.py (which only handles loading/cleaning/splitting)
"""

import os
import json
import time
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data_prep import (device, MAX_LENGTH, SOS_token, EOS_token, PAD_token, UNK_token,
                        tensorFromSentence)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "results.json")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")


# --------------------------------------------------------------------------
# Timing helpers (just for readable progress printouts)
# --------------------------------------------------------------------------
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # ignore_index=PAD_token -> padded positions don't contribute to the loss
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch_loss(dataloader, encoder, decoder, criterion):
    # same forward pass as training but no backward/optimizer step -- used
    # to track validation loss each epoch (with teacher forcing, for a fair
    # apples-to-apples comparison against training loss)
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train(train_dataloader, val_dataloader, encoder, decoder, n_epochs,
          learning_rate=0.001, print_every=5):
    start = time.time()
    train_losses, val_losses = [], []

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_token)   # PAD positions excluded from loss

    for epoch in range(1, n_epochs + 1):
        encoder.train(); decoder.train()
        tr_loss = train_epoch(train_dataloader, encoder, decoder,
                               encoder_optimizer, decoder_optimizer, criterion)
        encoder.eval(); decoder.eval()
        va_loss = eval_epoch_loss(val_dataloader, encoder, decoder, criterion)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if epoch % print_every == 0 or epoch == 1:
            print('%s (epoch %d/%d) train_loss=%.4f val_loss=%.4f' %
                  (timeSince(start, epoch / n_epochs), epoch, n_epochs, tr_loss, va_loss))

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, title, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch'); plt.ylabel('NLL Loss'); plt.title(title); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


# --------------------------------------------------------------------------
# Evaluation: greedy decoding, BLEU, qualitative samples
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    encoder.eval(); decoder.eval()
    input_tensor = tensorFromSentence(input_lang, sentence)   # UNK-safe conversion
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()
    if decoded_ids.dim() == 0:
        decoded_ids = decoded_ids.unsqueeze(0)

    decoded_words = []
    for idx in decoded_ids:
        if idx.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        if idx.item() == PAD_token:
            continue    # skip stray PAD predictions rather than printing them
        decoded_words.append(output_lang.index2word.get(idx.item(), '<UNK>'))
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=10, seed=20):
    import random
    rng = random.Random(seed)   # same seed (random_state=20) -> same samples shown every run
    sample = rng.sample(pairs, min(n, len(pairs)))
    for src, tgt in sample:
        print('>', src)
        print('=', tgt)
        output_words, _ = evaluate(encoder, decoder, src, input_lang, output_lang)
        print('<', ' '.join(w for w in output_words if w != '<EOS>'))
        print()


def compute_bleu(encoder, decoder, pairs, input_lang, output_lang):
    """Corpus-level BLEU-4 with smoothing (standard for short, sparse-reference NMT)."""
    references, hypotheses = [], []
    for src, tgt in pairs:
        output_words, _ = evaluate(encoder, decoder, src, input_lang, output_lang)
        hypotheses.append([w for w in output_words if w != '<EOS>'])
        references.append([tgt.split(' ')])
    smoothie = SmoothingFunction().method4
    return corpus_bleu(references, hypotheses, smoothing_function=smoothie)


# --------------------------------------------------------------------------
# Results shared across notebooks 01-05
# --------------------------------------------------------------------------
def count_parameters(*modules):
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


def perplexity(loss):
    """Perplexity = exp(average per-token NLL loss). Standard NMT metric --
    intuitively, the model's average 'branching factor' / how many words it
    was effectively choosing between at each step. Lower is better; 1.0 would
    mean perfect, fully-confident predictions."""
    return math.exp(loss)


def measure_inference_latency(encoder, decoder, pairs, input_lang, output_lang, n=100, warmup=5):
    """Average greedy-decoding latency per sentence, in milliseconds, measured
    on the current device. Excludes a few warmup calls so first-call overhead
    doesn't skew the average."""
    sample = pairs[:min(n, len(pairs))]

    for src, _ in sample[:warmup]:          # warmup, not timed
        evaluate(encoder, decoder, src, input_lang, output_lang)

    start = time.time()
    for src, _ in sample:
        evaluate(encoder, decoder, src, input_lang, output_lang)
    elapsed = time.time() - start

    avg_latency_ms = (elapsed / len(sample)) * 1000
    throughput = len(sample) / elapsed       # sentences/sec
    return avg_latency_ms, throughput


def save_result(model_name, metrics: dict):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    results[model_name] = metrics
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results for '{model_name}' -> {RESULTS_PATH}")


def load_results():
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)
