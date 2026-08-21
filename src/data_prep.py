"""
data_prep.py
------------
All data loading, cleaning, vocabulary building, and train/val/test splitting
is here. Every notebook just calls `load_split_data()` and gets ready-to-use DataLoaders + pair lists.

Special token convention used throughout this project:
    PAD = 0    padding (so batches of unequal-length sentences can be stacked)
    SOS = 1    start-of-sentence (decoder's first input)
    EOS = 2    end-of-sentence (marks where a sentence ends)
    UNK = 3    unknown/out-of-vocabulary word (for words seen at eval time
                that never appeared in the training vocabulary)

random_state = 20  is used for every random operation below:
subset sampling, train/val/test split, and DataLoader batch shuffling.
"""

import os
import re
import random
import unicodedata

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
RANDOM_STATE = 20          # reused everywhere for reproducibility
MAX_LENGTH   = 10          # max tokens per sentence (including EOS)
SUBSET_SIZE  = 8000        # sampled from the full 330k pair corpus 

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "deu.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=RANDOM_STATE):
    """Seed python random, numpy, and torch together so weight init and
    every downstream random.Random(seed) call are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
class Lang:
    """Maps words <-> indices for one language. Reserves indices 0-3 for
    PAD/SOS/EOS/UNK before any real words are added."""

    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "UNK": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.n_words = 4  # PAD, SOS, EOS, UNK already taken

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------
def unicodeToAscii(s):
    # strips accents, e.g. "über" -> "uber", so German umlauts don't
    # explode vocabulary size with accent-only variants
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # lowercase, split punctuation off words, drop anything that isn't a
    # letter or ! ? .
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


# --------------------------------------------------------------------------
# Loading + filtering
# --------------------------------------------------------------------------
def readLangs(path=DATA_PATH):
    # deu.txt lines: "English<TAB>German<TAB>attribution"
    # we translate German -> English, so pairs are reversed to [deu, eng]
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
    pairs = [list(reversed(p)) for p in pairs]

    input_lang = Lang('deu')
    output_lang = Lang('eng')
    return input_lang, output_lang, pairs


def filterPair(p):
    # keep only short sentences (both sides under MAX_LENGTH tokens)
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def prepareData(path=DATA_PATH, subset_size=SUBSET_SIZE, seed=RANDOM_STATE, verbose=True):
    """Load -> normalize -> filter by length -> take a fixed reproducible
    subset -> build vocabulary from that subset."""
    input_lang, output_lang, pairs = readLangs(path)
    if verbose:
        print("Read %d sentence pairs" % len(pairs))

    pairs = [p for p in pairs if filterPair(p)]
    if verbose:
        print("Trimmed to %d pairs (< %d tokens/sentence)" % (len(pairs), MAX_LENGTH))

    # fixed, reproducible subsample -- full pair corpus is too large to
    # train an RNN/attention model on a single CPU core in reasonable time
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:subset_size]
    if verbose:
        print("Sampled subset (random_state=%d): %d pairs" % (seed, len(pairs)))

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    if verbose:
        print("Vocab sizes -> %s: %d words | %s: %d words" %
              (input_lang.name, input_lang.n_words, output_lang.name, output_lang.n_words))

    return input_lang, output_lang, pairs


# --------------------------------------------------------------------------
# Train / Validation / Test split - 80 / 10 / 10, random_state = 20
# --------------------------------------------------------------------------
def train_val_test_split(pairs, val_frac=0.1, test_frac=0.1, seed=RANDOM_STATE):
    """80/10/10 split by default. Same seed as everything else, so every
    notebook gets the identical split -> fair comparison across models."""
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)

    n = len(idx)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]          # remaining 80% -> train

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    return train_pairs, val_pairs, test_pairs


# --------------------------------------------------------------------------
# Sentence, tensor conversion (UNK-safe)
# --------------------------------------------------------------------------
def indexesFromSentence(lang, sentence):
    # unseen words map to UNK instead of raising KeyError
    return [lang.word2index.get(word, UNK_token) for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def pairs_to_dataloader(pairs, input_lang, output_lang, batch_size=64, shuffle=True, seed=RANDOM_STATE):
    """Builds a padded (batch, MAX_LENGTH) tensor dataset. Sequences shorter
    than MAX_LENGTH are padded with PAD_token (0) on the right."""
    n = len(pairs)
    input_ids = np.full((n, MAX_LENGTH), PAD_token, dtype=np.int64)
    target_ids = np.full((n, MAX_LENGTH), PAD_token, dtype=np.int64)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)[:MAX_LENGTH - 1] + [EOS_token]
        tgt_ids = indexesFromSentence(output_lang, tgt)[:MAX_LENGTH - 1] + [EOS_token]
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                          torch.LongTensor(target_ids).to(device))

    if shuffle:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)                  # seeded batch shuffling
        sampler = RandomSampler(data, generator=gen)
        return DataLoader(data, sampler=sampler, batch_size=batch_size)
    return DataLoader(data, shuffle=False, batch_size=batch_size)


# --------------------------------------------------------------------------
# One call notebooks use to get everything ready
# --------------------------------------------------------------------------
def load_split_data(batch_size=64, verbose=True):
    """Single entry point every notebook calls. Returns vocabularies,
    raw pair lists (for qualitative eval/BLEU), and ready-to-train DataLoaders."""
    set_seed(RANDOM_STATE)
    input_lang, output_lang, pairs = prepareData(verbose=verbose)
    train_pairs, val_pairs, test_pairs = train_val_test_split(pairs)  # 80/10/10

    if verbose:
        print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    train_loader = pairs_to_dataloader(train_pairs, input_lang, output_lang, batch_size=batch_size, shuffle=True)
    val_loader = pairs_to_dataloader(val_pairs, input_lang, output_lang, batch_size=batch_size, shuffle=False)
    test_loader = pairs_to_dataloader(test_pairs, input_lang, output_lang, batch_size=batch_size, shuffle=False)

    return {
        "input_lang": input_lang, "output_lang": output_lang,
        "train_pairs": train_pairs, "val_pairs": val_pairs, "test_pairs": test_pairs,
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
    }
