"""Sanity-check predict.py on hand-written messages (not from the dataset) and save the
results to outputs/sample_predictions.json.

Usage:  python src/check_predictions.py
"""
import json

from config import OUTPUTS_DIR
from predict import predict

SAMPLES = [
    ("Congratulations! You have won a GBP 1000 cash prize. Call 09061234567 now to claim, T&C apply", "SPAM"),
    ("URGENT: your mobile number has been selected for a free iPhone. Reply YES to 80085 to claim", "SPAM"),
    ("FREE entry into our weekly draw! Text WIN to 85233 for a chance to win a holiday worth 500 pounds", "SPAM"),
    ("Your account has been suspended, click http://bit.ly/secure-login to verify your details", "SPAM"),
    ("Hey, are we still meeting for lunch tomorrow at 1?", "HAM"),
    ("Can you pick up some milk on your way home please", "HAM"),
    ("Sorry I missed your call, I was in a meeting. Will call you back tonight", "HAM"),
    ("Happy birthday bro! Have a great day, see you at the party", "HAM"),
]


def main():
    preds = predict([msg for msg, _ in SAMPLES])
    rows = []
    for (msg, expected), (label, prob) in zip(SAMPLES, preds):
        rows.append({"message": msg, "expected": expected, "predicted": label,
                     "spam_probability": round(prob, 4), "correct": label == expected})
        print(f"{'OK ' if label == expected else 'BAD'} {label:4s} p={prob:.4f}  {msg}")
    correct = sum(r["correct"] for r in rows)
    print(f"{correct}/{len(rows)} correct")
    (OUTPUTS_DIR / "sample_predictions.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
