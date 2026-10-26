"""Loading, cleaning, tokenizing, padding and splitting the SMS Spam Collection."""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_FILE, MAX_LEN, SEED, TEST_SIZE, TOKENIZER_FILE,
                    VAL_SIZE, VOCAB_SIZE)

PAD, OOV = "<pad>", "<oov>"

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\S+@\S+")
_CURRENCY = re.compile(r"[£$€]")
_NUMBER = re.compile(r"\d+")
_NON_ALNUM = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def load_dataset(path=DATA_FILE):
    """Read the tab-separated UCI file into a DataFrame with label/text/target columns."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            label, text = line.rstrip("\n").split("\t", 1)
            rows.append((label, text))
    df = pd.DataFrame(rows, columns=["label", "text"])
    df["target"] = (df["label"] == "spam").astype(int)
    return df


def clean_text(text):
    """Lowercase and normalise an SMS; URLs, numbers etc. become placeholder words."""
    text = text.lower()
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _CURRENCY.sub(" currencytoken ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


class SimpleTokenizer:
    """Word-level tokenizer: keeps the most frequent words seen in the training set."""

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word_index = {}

    def fit(self, texts):
        counts = Counter(w for t in texts for w in t.split())
        most_common = [w for w, _ in counts.most_common(self.vocab_size - 2)]
        self.word_index = {PAD: 0, OOV: 1}
        self.word_index.update({w: i + 2 for i, w in enumerate(most_common)})
        return self

    def texts_to_sequences(self, texts):
        return [[self.word_index.get(w, 1) for w in t.split()] for t in texts]

    def save(self, path=TOKENIZER_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "word_index": self.word_index}, f)

    @classmethod
    def load(cls, path=TOKENIZER_FILE):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["vocab_size"])
        tok.word_index = data["word_index"]
        return tok


def pad(sequences, max_len=MAX_LEN):
    """Pre-pad with zeros (so the RNN's last steps see real words) and truncate long messages."""
    out = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        if seq:
            out[i, -len(seq):] = seq
    return out


def prepare_data(save_tokenizer=True):
    """Clean -> stratified 70/15/15 split -> fit tokenizer on train only -> pad."""
    df = load_dataset()
    df["clean"] = df["text"].map(clean_text)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["target"], random_state=SEED)
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=train_df["target"],
        random_state=SEED)

    tokenizer = SimpleTokenizer().fit(train_df["clean"])
    if save_tokenizer:
        tokenizer.save()

    def encode(frame):
        return pad(tokenizer.texts_to_sequences(frame["clean"])), frame["target"].to_numpy()

    X_train, y_train = encode(train_df)
    X_val, y_val = encode(val_df)
    X_test, y_test = encode(test_df)
    return {
        "df": df, "tokenizer": tokenizer,
        "train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test),
        "test_text": test_df["text"].tolist(),
    }


def summarize():
    """Print dataset / split statistics (used for NOTES.md and the report)."""
    data = prepare_data(save_tokenizer=False)
    df = data["df"]
    lengths = df["clean"].str.split().map(len)
    print(f"Total messages: {len(df)}")
    print(df["label"].value_counts().to_string())
    print(f"Spam share: {df['target'].mean():.4f}")
    print(f"Tokens/message: mean={lengths.mean():.1f} median={lengths.median():.0f} "
          f"p95={lengths.quantile(0.95):.0f} max={lengths.max()}")
    print(f"Share of messages with <= MAX_LEN tokens: {(lengths <= MAX_LEN).mean():.4f}")
    print(f"Distinct words (full corpus): {len(set(w for t in df['clean'] for w in t.split()))}")
    for name in ("train", "val", "test"):
        y = data[name][1]
        print(f"{name}: n={len(y)} spam={int(y.sum())} ham={int(len(y) - y.sum())}")


if __name__ == "__main__":
    summarize()
