"""Build ppt/RNN_SMS_Spam_Detection.pptx (title slide + 6 content slides).

Only native shapes, text and tables are used; the two pictures are the real graphs from
outputs/. All numbers are read from outputs/*.json and models/best_config.json.

Usage:  python ppt/build_ppt.py
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ppt" / "RNN_SMS_Spam_Detection.pptx"
OUTPUTS = ROOT / "outputs"

NAVY = RGBColor(0x1E, 0x27, 0x61)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
TINT = RGBColor(0xF1, 0xF4, 0xFA)
ACCENT = RGBColor(0xD9, 0x4E, 0x1F)      # spam / alert accent (orange-red)
GREEN = RGBColor(0x1B, 0x7F, 0x4C)       # ham chip
INK = RGBColor(0x1F, 0x23, 0x33)
MUTED = RGBColor(0x5A, 0x60, 0x70)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"

test = json.loads((OUTPUTS / "test_metrics.json").read_text())
best = json.loads((ROOT / "models" / "best_config.json").read_text())
hist = json.loads((ROOT / "models" / "best_history.json").read_text())
preds = json.loads((OUTPUTS / "sample_predictions.json").read_text())
(TN, FP), (FN, TP) = test["confusion_matrix"]
ACC, PREC, REC, F1 = (f"{test['accuracy'] * 100:.2f}%", f"{test['precision']:.4f}",
                      f"{test['recall']:.4f}", f"{test['f1']:.4f}")
EPOCH = best["weights_from_epoch"]
CW = best["class_weight"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullets=False, space=8, name=None):
    """content: str or list of str / (str, {overrides}) paragraphs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    items = content if isinstance(content, list) else [content]
    for i, item in enumerate(items):
        t, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space", space))
        if o.get("bullet", bullets):
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.3)))
            pPr.set("indent", str(-Inches(0.25)))
            bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar",
                                 {"char": "•"})
            pPr.append(bu)
        r = p.add_run()
        r.text = t
        f = r.font
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.name = o.get("font", font)
        f.color.rgb = o.get("color", color)
    return box


def card(slide, x, y, w, h, fill=TINT, line=None, radius=0.08, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line:
        shp.line.color.rgb = line
        shp.line.width = Pt(1.25)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def title(slide, t, color=NAVY):
    text(slide, 0.6, 0.45, 12.1, 0.9, t, size=36, bold=True, font=HEAD, color=color,
         anchor=MSO_ANCHOR.MIDDLE, name="Title")


def chip(slide, x, y, label, color):
    c = card(slide, x, y, 1.0, 0.38, fill=color, radius=0.5)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(13), True, BODY, WHITE


def stat(slide, x, y, w, value, label, value_color=NAVY, label_color=MUTED, fill=TINT, h=1.45):
    card(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.2, y + 0.12, w - 0.4, 0.75, value, size=36, bold=True, font=HEAD,
         color=value_color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.2, y + 0.85, w - 0.4, h - 0.95, label, size=14, color=label_color,
         align=PP_ALIGN.CENTER)


def table(slide, x, y, w, rows, col_w, font_size=14, row_h=0.4):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * len(rows)))
    tbl = shape.table
    for ci, cw in enumerate(col_w):
        tbl.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if ri == 0 else (TINT if ri % 2 == 0 else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = val
            r.font.size, r.font.name = Pt(font_size), BODY
            r.font.bold = ri == 0 or ci == 0
            r.font.color.rgb = WHITE if ri == 0 else INK
    return shape


# ---------------------------------------------------------------- title slide
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, 0.9, 1.5, 11.5, 0.5, "DEEP LEARNING STUDY PROJECT", size=16, bold=True, color=ICE)
text(s, 0.9, 2.05, 11.5, 1.9, "SMS Spam Detection using Recurrent Neural Networks",
     size=44, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 3.95, 11.5, 0.6, "Classifying SMS messages as spam or ham with an "
     "Embedding → GRU → Dense → Sigmoid network in TensorFlow/Keras",
     size=18, color=ICE)
text(s, 0.9, 5.35, 7.0, 1.3, [("Name of the Student  |  23EG110XXX  |  CSE(DS)", {"bold": True}),
     "Under the guidance of Dr P Rajeshwari", "Department of CSE (Data Science), Anurag University"],
     size=16, color=WHITE, space=4)
s.notes_slide.notes_text_frame.text = (
    "Title slide. Replace the student name and roll number placeholders before presenting.")

# ---------------------------------------------------------------- 1. problem statement
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Problem Statement")
text(s, 0.6, 1.6, 6.2, 4.8, [
    "SMS spam (fake prizes, premium-rate numbers, phishing links) is a daily nuisance and a "
    "security risk",
    "Keyword filters are easy to bypass when spammers change their wording",
    "Goal: learn spam patterns from real labelled messages with a Recurrent Neural Network",
    "Output: a model that labels any SMS as SPAM or HAM, plus a prediction script",
    "Targets: test accuracy ≥ 95% and spam F1-score ≥ 0.90",
], size=18, bullets=True, space=14)
card(s, 7.3, 1.6, 5.4, 2.25)
chip(s, 7.6, 1.85, "SPAM", ACCENT)
text(s, 7.6, 2.35, 4.9, 1.4, "“Free entry in 2 a wkly comp to win FA Cup final tkts 21st "
     "May 2005. Text FA to 87121 to receive entry question…”", size=15, color=INK)
card(s, 7.3, 4.1, 5.4, 1.75)
chip(s, 7.6, 4.35, "HAM", GREEN)
text(s, 7.6, 4.85, 4.9, 0.9, "“Ok lar... Joking wif u oni...”", size=15, color=INK)
text(s, 7.3, 6.0, 5.4, 0.5, "Real examples from the UCI SMS Spam Collection", size=12,
     color=MUTED)
s.notes_slide.notes_text_frame.text = (
    "Spam is a binary text-classification problem: every message is either spam or ham.")

# ---------------------------------------------------------------- 2. dataset
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Dataset Used")
stat(s, 0.6, 1.55, 3.85, "5,574", "SMS messages – UCI SMS Spam Collection")
stat(s, 4.74, 1.55, 3.85, "4,827", "ham (legitimate) – 86.6%", value_color=GREEN)
stat(s, 8.88, 1.55, 3.85, "747", "spam – 13.4% (imbalanced)", value_color=ACCENT)
text(s, 0.6, 3.35, 6.3, 3.6, [
    ("Preprocessing", {"bold": True, "size": 20, "font": HEAD, "color": NAVY}),
    ("Lowercase; URLs, e-mails, currency and numbers → placeholder words", {"bullet": True}),
    ("Remove punctuation, tokenize (5,000-word vocabulary)", {"bullet": True}),
    ("Pad / truncate to 50 tokens (covers 98.7% of messages)", {"bullet": True}),
    ("Class weights to handle the imbalance", {"bullet": True}),
], size=17, space=10)
text(s, 7.3, 3.35, 5.4, 0.5, "Stratified split (70 / 15 / 15)", size=20, bold=True, font=HEAD,
     color=NAVY)
table(s, 7.3, 3.95, 5.4, [["Split", "Messages", "Spam"], ["Training", "3,901", "523"],
                          ["Validation", "836", "112"], ["Test", "837", "112"]],
      [2.0, 1.8, 1.6], font_size=16, row_h=0.5)
text(s, 7.3, 6.1, 5.4, 0.5, "Tokenizer fitted on the training split only (no leakage)",
     size=12, color=MUTED)
s.notes_slide.notes_text_frame.text = (
    "Downloaded automatically from the UCI Machine Learning Repository (Almeida & Gomez Hidalgo, 2011).")

# ---------------------------------------------------------------- 3. architecture
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Model Architecture")
layers = [("Input", "50 word ids", "(50)", "0"),
          ("Embedding", "5,000 × 64", "(50, 64)", "320,000"),
          ("GRU", "64 units, dropout 0.3", "(64)", "24,960"),
          ("Dense", "32 ReLU + dropout 0.3", "(32)", "2,080"),
          ("Output", "1 unit, Sigmoid", "(1)", "33")]
bw, gap, x0, y0 = 2.05, 0.45, 0.6, 1.9
for i, (name, desc, shape, params) in enumerate(layers):
    x = x0 + i * (bw + gap)
    hl = name == "GRU"
    c = card(s, x, y0, bw, 1.7, fill=NAVY if hl else TINT, name=f"Layer {name}")
    text(s, x + 0.1, y0 + 0.22, bw - 0.2, 0.55, name, size=20, bold=True, font=HEAD,
         color=WHITE if hl else NAVY, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, y0 + 0.85, bw - 0.2, 0.7, desc, size=14, color=ICE if hl else MUTED,
         align=PP_ALIGN.CENTER)
    text(s, x, y0 + 1.85, bw, 0.4, f"shape {shape}", size=14, color=INK, align=PP_ALIGN.CENTER)
    text(s, x, y0 + 2.22, bw, 0.4, f"{params} params", size=14, color=MUTED, align=PP_ALIGN.CENTER)
    if i < len(layers) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.07), Inches(y0 + 0.65),
                               Inches(gap - 0.14), Inches(0.4))
        a.fill.solid()
        a.fill.fore_color.rgb = ACCENT
        a.line.fill.background()
card(s, 0.6, 4.9, 12.1, 1.75)
text(s, 0.9, 5.05, 11.5, 1.5, [
    ("Spam probability ≥ 0.5 → SPAM, otherwise HAM", {"bold": True, "color": NAVY}),
    (f"{best['params']:,} trainable parameters • the GRU reads the message word by word and "
     "its gates decide what to remember", {}),
    ("Chosen from 6 attempts (SimpleRNN, LSTM, GRU) by the best validation F1-score", {}),
], size=17, space=6)
s.notes_slide.notes_text_frame.text = (
    "Embedding -> recurrent layer -> Dense -> Sigmoid. The GRU is a gated recurrent layer; the best "
    "pure SimpleRNN reached 98.57% test accuracy (F1 0.9455).")

# ---------------------------------------------------------------- 4. hyperparameters
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Hyperparameters")
table(s, 0.6, 1.5, 7.2, [
    ["Hyperparameter", "Value"],
    ["Vocabulary size / sequence length", "5,000 words / 50 tokens"],
    ["Embedding dimension", "64"],
    ["Recurrent layer", "GRU, 64 units"],
    ["Dense layer", "32 units, ReLU"],
    ["Dropout", "0.3"],
    ["Optimizer", "Adam, lr 0.001, clipnorm 1.0"],
    ["Loss", "Binary cross-entropy"],
    ["Batch size", "32"],
    ["Epochs", f"max 40 → stopped at {best['epochs_run']}, best epoch {EPOCH}"],
    ["Early stopping", "val_loss, patience 5, restore best"],
    ["Class weights (ham / spam)", f"{CW['0']:.3f} / {CW['1']:.3f}"],
], [3.4, 3.8], font_size=14, row_h=0.43)
card(s, 8.3, 1.5, 4.4, 5.16)
text(s, 8.6, 1.7, 3.9, 4.8, [
    ("How it was tuned", {"bold": True, "size": 20, "font": HEAD, "color": NAVY, "space": 12}),
    ("Baseline SimpleRNN overfitted (100% train accuracy, rising val loss)", {"bullet": True}),
    ("Added dropout and early stopping", {"bullet": True}),
    ("Lower learning rate and gradient clipping", {"bullet": True}),
    ("Tried LSTM and GRU cells", {"bullet": True}),
    ("Softened the spam class weight", {"bullet": True}),
    ("Best model picked on validation data, not test", {"bullet": True, "bold": True}),
], size=16, space=10)
s.notes_slide.notes_text_frame.text = (
    "Six training runs in total; the full results table is in NOTES.md.")

# ---------------------------------------------------------------- 5. results
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Results: Accuracy and Loss")
iw = 6.0
for i, (img, cap) in enumerate([
        ("accuracy_graph.png", f"Dashed line: weights from epoch {EPOCH} restored by early stopping"),
        ("loss_graph.png", f"Validation loss rises after epoch {EPOCH}: early stopping prevents overfitting")]):
    x = 0.6 + i * (iw + 0.13)
    s.shapes.add_picture(str(OUTPUTS / img), Inches(x), Inches(1.4), width=Inches(iw))
    text(s, x, 1.4 + iw * 0.6 + 0.02, iw, 0.4, cap, size=12, color=MUTED, align=PP_ALIGN.CENTER)
mw = 2.9
for i, (v, l) in enumerate([(ACC, "Test accuracy"), (PREC, "Precision (spam)"),
                            (REC, "Recall (spam)"), (F1, "F1-score (spam)")]):
    x = 0.6 + i * (mw + 0.17)
    card(s, x, 5.55, mw, 1.3)
    text(s, x + 0.1, 5.6, mw - 0.2, 0.7, v, size=30, bold=True, font=HEAD, color=NAVY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.1, 6.28, mw - 0.2, 0.45, l, size=14, color=MUTED, align=PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = (
    f"Weights from epoch {EPOCH} were restored by early stopping (dashed line). After that the "
    "training loss keeps falling while the validation loss rises, which is overfitting. "
    f"Test confusion matrix: TN {TN}, FP {FP}, FN {FN}, TP {TP}.")

# ---------------------------------------------------------------- 6. conclusion
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
title(s, "Conclusion", color=WHITE)
correct = sum(p["correct"] for p in preds)
text(s, 0.6, 1.6, 7.3, 5.0, [
    f"The RNN (GRU) model detects SMS spam with {ACC} test accuracy and spam F1 {F1}",
    f"Only {FP} of {TN + FP} legitimate messages was wrongly flagged; {TP} of {TP + FN} spam caught",
    "Class weights, dropout and early stopping fixed the baseline's overfitting",
    f"The prediction script labelled {correct} of {len(preds)} new test messages correctly",
    "Future work: larger multilingual data, pre-trained embeddings, mobile / web deployment",
], size=18, color=WHITE, bullets=True, space=16)
stat(s, 8.5, 1.7, 4.2, ACC, "test accuracy (837 messages)", value_color=WHITE,
     label_color=ICE, fill=RGBColor(0x2B, 0x36, 0x7C), h=1.75)
stat(s, 8.5, 3.75, 4.2, F1, "spam F1-score (target 0.90)", value_color=WHITE,
     label_color=ICE, fill=RGBColor(0x2B, 0x36, 0x7C), h=1.75)
s.notes_slide.notes_text_frame.text = "All numbers come from the final run recorded in NOTES.md."

prs.save(OUT)
print("Saved", OUT)
