# NOTES – decisions, assumptions and experiment log

All numbers below were copied from real runs (`outputs/experiments_results.json`,
`outputs/test_metrics.json`, `models/best_history.json`). Nothing is estimated.

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| Host Python | 3.12.10 |
| Training runtime | Docker image `tensorflow/tensorflow:2.21.0` (+ requirements.txt), CPU only |
| Libraries in container | TensorFlow 2.21.0, scikit-learn 1.9.1, pandas 3.1.0, matplotlib 3.11.2 |
| LibreOffice | `C:\Program Files\LibreOffice\program\soffice.exe` (used for PDF checks) |

**Problem found:** Windows **Smart App Control is ON** on this PC (`SmartAppControlState = On`).
It blocks unsigned native DLLs: `import tensorflow` failed with *"DLL load failed … An
Application Control policy has blocked this file"*, both in the global Python 3.12 and
in a fresh `.venv`, and in the venv even `pandas` (`pandas._libs.interval`) was blocked.
The global Python also had a numpy 2.5 / scikit-learn / pandas binary mismatch.

**Approaches tried:** (1) global Python 3.12 → blocked; (2) clean `.venv` with fresh
pip wheels → still blocked; (3) official TensorFlow Linux image in Docker Desktop
(already running) → **works**. I did not disable Smart App Control, because it is an OS
security feature (and it cannot be re-enabled without reinstalling Windows).
`Dockerfile` in the repo reproduces the environment. On a normal PC or Colab,
`pip install -r requirements.txt` is enough.

The host `.venv` (git-ignored) is still used for non-TensorFlow work (python-docx and
python-pptx for the report and slides).

## 2. Dataset

* **Source:** UCI Machine Learning Repository, *SMS Spam Collection* (id 228),
  `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`. It downloaded
  on the first attempt, so no mirror was needed.
* Stored as `data/SMSSpamCollection.tsv` (original tab-separated file, unchanged) and
  `data/readme.txt` (original UCI readme, which holds the citation and licence information).
* SHA-256 of `SMSSpamCollection.tsv`:
  `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`
* **5,574 messages: 4,827 ham (86.6 %) and 747 spam (13.4 %).** The classes are imbalanced.
* Cleaned length: mean 16.5 tokens, median 13, 95th percentile 35, max 190;
  98.73 % of messages have ≤ 50 tokens. Distinct words (full corpus): 7,730.

## 3. Preprocessing (src/data_prep.py)

1. Lowercase.
2. Replace URLs → `urltoken`, e-mail addresses → `emailtoken`, `£ $ €` → `currencytoken`,
   digit runs → `numtoken`. *Assumption:* phone numbers, short codes and amounts are a
   strong spam signal, so they are kept as placeholder words instead of being deleted.
3. Remove every other non-letter character and collapse whitespace.
4. **Stratified split 70 / 15 / 15** (`random_state=42`):
   train 3,901 (523 spam), validation 836 (112 spam), test 837 (112 spam).
5. Word tokenizer **fitted on the training split only** (no leakage). Vocabulary = 5,000
   (index 0 = padding, 1 = out-of-vocabulary). Saved to `models/tokenizer.json`.
6. Sequences **pre-padded / post-truncated to 50 tokens**. Pre-padding means the
   recurrent layer's last steps see real words. The Embedding uses `mask_zero=True`.

*Assumption:* I used a small custom tokenizer instead of `keras.preprocessing.text.Tokenizer`,
because that class is deprecated or removed in Keras 3. The custom one is 20 lines and
easy to explain.

## 4. Class imbalance

`sklearn.utils.class_weight.compute_class_weight("balanced")` on the training labels gives
ham = 0.5774 and spam = 3.7294. Attempts A1–A5 used these weights. A6 halves the spam
weight (spam = 1.8647), because the remaining validation errors were mostly ham flagged
as spam (false positives).

## 5. Self-evaluation loop – every attempt

Targets: test accuracy ≥ 95 % **and** spam F1 ≥ 0.90. I also checked every run for
overfitting. Selection rule (fixed before comparing models): **highest validation spam-F1**,
ties broken by validation accuracy, then by lower validation loss. Test scores were *not*
used to choose the model.

Architecture for every run: `Embedding → recurrent layer → Dense(ReLU) → Dropout → Dense(1, sigmoid)`,
Adam, binary cross-entropy, batch size 32, seed 42, deterministic TF ops.
"Emb/RNN/Dense" = embedding dimension / recurrent units / dense units.
ES = EarlyStopping(monitor=val_loss, restore_best_weights=True).

| # | Run | Cell, size | Regularisation / changes | Epochs run (weights from) | Val acc | Val P | Val R | **Val F1** | Val loss | Test acc | Test P | Test R | Test F1 | Test CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A1_baseline | SimpleRNN 64/64/32 | none, lr 1e-3, 15 epochs, no ES | 15 (15) | 0.9880 | 0.9554 | 0.9554 | 0.9554 | 0.0600 | 0.9857 | 0.9808 | 0.9107 | 0.9444 | [[723,2],[10,102]] |
| 2 | A2_simplernn_dropout_es | SimpleRNN 64/64/32 | dropout 0.3, ES patience 3 | 4 (1) | 0.9749 | 0.8527 | 0.9821 | 0.9129 | 0.1166 | 0.9618 | 0.8125 | 0.9286 | 0.8667 | [[701,24],[8,104]] |
| 3 | A3_simplernn_small_lr | SimpleRNN 32/32/16 | dropout 0.3, lr 5e-4, clipnorm 1.0, ES patience 5 | 18 (13) | 0.9821 | 0.9076 | 0.9643 | 0.9351 | 0.0661 | 0.9857 | 0.9630 | 0.9286 | 0.9455 | [[721,4],[8,104]] |
| 4 | A4_lstm | LSTM 64/64/32 | dropout 0.3, lr 1e-3, clipnorm 1.0, ES patience 5 | 9 (4) | 0.9868 | 0.9316 | 0.9732 | 0.9520 | 0.0421 | 0.9892 | 0.9905 | 0.9286 | 0.9585 | [[724,1],[8,104]] |
| 5 | A5_gru | GRU 64/64/32 | as A4 | 9 (4) | 0.9916 | 0.9565 | 0.9821 | 0.9692 | 0.0318 | 0.9916 | 0.9907 | 0.9464 | 0.9680 | [[724,1],[6,106]] |
| 6 | **A6_gru_softer_weights** | GRU 64/64/32 | as A5 + spam class weight × 0.5 | 9 (4) | **0.9928** | **0.9732** | **0.9732** | **0.9732** | **0.0251** | **0.9904** | **0.9906** | **0.9375** | **0.9633** | **[[724,1],[7,105]]** |

"Val loss" is the validation loss of the saved weights.

**What happened in each attempt**

* **A1:** passes the targets, but **overfits**: train accuracy reaches 1.0000 and train loss
  falls to 0.0002, while validation loss bottoms out at 0.0531 (epoch 5) and climbs back to
  0.0600. Fixing the overfitting was required.
* **A2:** adding dropout and early stopping (patience 3) was too aggressive. Validation loss
  was very noisy, so training stopped at epoch 4 with epoch-1 weights. 24 false positives on
  test. Worse.
* **A3:** a smaller SimpleRNN with lower learning rate, gradient clipping and patience 5 is
  stable (weights from epoch 13) and passes the targets (test F1 0.9455). It is the best
  pure SimpleRNN run.
* **A4 / A5:** gated cells (LSTM, GRU) reach a clearly lower validation loss. GRU was best.
* **A6:** a softer spam weight reduced validation false positives from 5 to 3. Best on
  every validation measure → **selected**.
* Honest note: A5 scored slightly *higher on the test set* (F1 0.9680 vs 0.9633,
  one more spam caught). The rule is to select by validation, so A6 stays the final model.

**Overfitting check for the final model (A6):** training continued to epoch 9, where train
accuracy hit 1.0000 while validation loss rose from 0.0251 to 0.0423. Early stopping
restored the **epoch-4 weights**, where train/val accuracy = 0.9974 / 0.9928 and train/val
loss = 0.0113 / 0.0251. The gap is small, so the saved model is not overfitted. The graphs
mark the restored epoch with a dashed line.

**Deviation from the requested architecture (assumption):** the brief described
`Embedding → SimpleRNN → Dense → Sigmoid` and also allowed LSTM/GRU in the improvement
loop. The validation-based selection picked a **GRU** (a gated recurrent layer). The
model is otherwise the same simple stack. The best SimpleRNN (A3: test accuracy 98.57 %,
F1 0.9455) also meets the targets. To use it, select A3 in `src/evaluate.py`.

## 6. Final model (A6_gru_softer_weights) – test set (837 messages, 112 spam)

| Metric | Value |
|---|---|
| Accuracy | **0.9904** (99.04 %) |
| Precision (spam) | **0.9906** |
| Recall (spam) | **0.9375** |
| F1 (spam) | **0.9633** |
| Confusion matrix | TN 724, FP 1, FN 7, TP 105 |
| Macro-avg F1 | 0.9789 |
| Parameters | 347,073 (Embedding 320,000; GRU 24,960; Dense 2,080; Output 33) |
| Training | 9 epochs run, weights from epoch 4, 37.2 s on CPU |

The reloaded `models/best_model.keras` reproduces these test numbers exactly (`src/evaluate.py`).

## 7. Prediction script check (src/predict.py)

Eight new messages, written for this check (none are from the dataset):

| Message | Expected | Output (spam probability) |
|---|---|---|
| Congratulations! You have won a GBP 1000 cash prize. Call 09061234567 now to claim, T&C apply | spam | SPAM (1.0000) |
| URGENT: your mobile number has been selected for a free iPhone. Reply YES to 80085 to claim | spam | SPAM (0.9999) |
| FREE entry into our weekly draw! Text WIN to 85233 for a chance to win a holiday worth 500 pounds | spam | SPAM (0.9999) |
| Your account has been suspended, click http://bit.ly/secure-login to verify your details | spam | SPAM (0.9806) |
| Hey, are we still meeting for lunch tomorrow at 1? | ham | HAM (0.0000) |
| Can you pick up some milk on your way home please | ham | HAM (0.0000) |
| Sorry I missed your call, I was in a meeting. Will call you back tonight | ham | HAM (0.0001) |
| Happy birthday bro! Have a great day, see you at the party | ham | HAM (0.0000) |

All 8 are correct.

### Numbers exactly as printed in the report and PPT

Final model A6, test set: accuracy **99.04%** (0.9904), precision **0.9906**, recall **0.9375**,
F1 **0.9633**, TN 724 / FP 1 / FN 7 / TP 105 (829 of 837 correct; 106 flagged as spam).
Validation at the restored epoch 4: accuracy **99.28%** (0.9928), loss **0.0251**; validation loss
at epoch 9 **0.0423**. Class weights in the report/PPT are rounded to 3 decimals:
ham **0.577**, spam **1.865**. Learning rate of A3 = 5e-4 = **0.0005**.
Other attempts' test accuracy as percentages: A1 98.57%, A2 96.18%, A3 98.57%, A4 98.92%,
A5 99.16%, A6 99.04%.

## 8. Report (report/RNN_SMS_Spam_Detection_Report.docx)

Built by `report/build_report.py` from a **copy** of `Raja/Sample report.docx` (the file in the
parent folder has a space in its name, not an underscore). The script reads every number from
the `outputs/*.json` files.

* **Kept unchanged:** cover page layout, Anurag University logo, all cover placeholders, the
  INDEX table, page size and margins (A4, 1"), and `styles.xml`, `numbering.xml`,
  `settings.xml`, `theme1.xml` and `fontTable.xml`. The XML content is identical; only the
  XML declaration's quote style differs after saving. The same section order and the same
  11 heading paragraphs (copied verbatim) are used. Each section starts on a new page, as in
  the sample.
* **Only text was replaced.** Every new paragraph, bullet and table is a deep copy of a
  paragraph or of the INDEX table ("Table Grid") from the sample. The report uses Times New
  Roman only, sizes 12/14/16/18 pt only, and the sample's own list numbering only. I checked
  this automatically: no font, size, style, numbering or table style was added.
* **Assumption – text colour:** the sample shows its example text in red (`EE0000`), with
  instructions such as "Find sample text below". I treated the red as the template's marker
  for "sample, replace me", so the new body text is black (`000000`, a colour the sample
  already uses for headings and the Methodology text). No new colour was introduced. To get
  red back, remove the `strip_red` calls in `build_report.py`.
* **Removed:** the template's instruction lines ("Find sample below", "Mention the
  architecture…", etc.) and the old project's screenshots (R code, Shiny UI, scatter plot).
  The orphaned image files were dropped from the package. Only the original logo and the three
  real graphs from `outputs/` remain (checked by MD5).
* **Code excerpts** (Model Performance → "Program:") are text inside sample-style tables, not
  screenshots, so no new images were generated.
* **Graphs** go in the GRAPHS section with bold labels in the sample's "Output:" label style
  ("Figure 1: …"). Width is 5.6" for the curves (fits inside the 0.5" list indent) and 4.2" for
  the confusion matrix.
* **INDEX page numbers** were empty in the sample. I filled them from the rendered PDF
  (3, 4, 5, 6, 7, 8, 9, 10, 15, 13, 16), then re-rendered and confirmed they did not move.
  Rows 9/10 keep the sample's order (Analysis & Results before Graphs), even though the body
  has Graphs first, as in the sample.
* The document properties (File → Info) still show the sample's original author metadata
  ("harshith gadwala"), because I left the placeholders untouched. Change this in Word if
  needed.
* Verification: rendered with LibreOffice → 16 pages, inspected visually; no leftover sample
  text (searched for Hyderabad, Villa, Shiny, Ranger, "Find sample", etc.); the docx
  schema validator passed.

## 9. Presentation (ppt/RNN_SMS_Spam_Detection.pptx)

Built by `ppt/build_ppt.py` with python-pptx (pptxgenjs was not installed). 16:9, 7 slides:
title, then exactly 1 Problem Statement, 2 Dataset Used, 3 Model Architecture (native-shape
diagram), 4 Hyperparameters (table), 5 Results – accuracy and loss graphs (real PNGs from
`outputs/`), 6 Conclusion. Design: navy/white with an orange accent, Cambria headings and
Calibri body (fonts that ship with Office). There are no logos, icons or stock images; the
only pictures are the two graphs. The title slide uses the same placeholders as the report.
Verification: rendered to PDF and checked every slide for overflow and overlap; the pptx
validator passed; every metric matches this file.
