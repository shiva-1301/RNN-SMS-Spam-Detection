"""Classify custom SMS messages as SPAM or HAM with the saved best model.

Usage:
    python src/predict.py "WINNER!! Claim your free prize now, call 09061701461"
    python src/predict.py            # interactive: type messages, empty line to quit
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from config import BEST_MODEL_FILE, THRESHOLD
from data_prep import SimpleTokenizer, clean_text, pad

_model = None
_tokenizer = None


def predict(messages):
    """Return a list of (label, spam_probability) tuples."""
    global _model, _tokenizer
    if _model is None:
        _model = tf.keras.models.load_model(BEST_MODEL_FILE)
        _tokenizer = SimpleTokenizer.load()
    X = pad(_tokenizer.texts_to_sequences([clean_text(m) for m in messages]))
    probs = _model.predict(X, verbose=0).ravel()
    return [("SPAM" if p >= THRESHOLD else "HAM", float(p)) for p in probs]


def main():
    messages = sys.argv[1:]
    if messages:
        for msg, (label, prob) in zip(messages, predict(messages)):
            print(f"{label:4s}  (spam probability {prob:.4f})  <- {msg}")
        return
    print("Type an SMS and press Enter (empty line to quit).")
    while True:
        msg = input("> ").strip()
        if not msg:
            break
        label, prob = predict([msg])[0]
        print(f"{label}  (spam probability {prob:.4f})")


if __name__ == "__main__":
    main()
