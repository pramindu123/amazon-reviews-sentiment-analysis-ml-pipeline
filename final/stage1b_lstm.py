"""
=====================================================================
 STAGE 1b — DEEP LEARNING BASELINE (LSTM)
 Machine Learning-Based Sentiment Analysis of Amazon Fine Food Reviews
 Module: ITC 4378 - Big Data Analytics
---------------------------------------------------------------------
 This is the deep-learning counterpart to the classical models in
 stage1_pipeline.py. It deliberately reuses the SAME data and the
 SAME evaluation idea so the comparison is fair:

   * reads sample_stage1.parquet  (produced by Stage 1)
   * trains on a BALANCED set, evaluates on a REALISTIC (imbalanced)
     test set -> honest metrics, directly comparable to Stage 1
   * reports accuracy / precision / recall / F1 in the same shape

 Why LSTM lives in its own script (not inside the Spark pipeline):
   - it needs padded INTEGER SEQUENCES, not TF-IDF sparse vectors
   - it trains in Keras/TensorFlow, not Spark MLlib
   - it benefits from a GPU (enable one in Colab: Runtime > Change
     runtime type > GPU)
=====================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ------------------------------------------------------------------ #
#  CONFIGURATION
# ------------------------------------------------------------------ #
PARQUET_PATH = "sample_stage1.parquet"   # output of Stage 1
SEED         = 42
MAX_WORDS    = 20_000    # vocabulary cap (most frequent words kept)
MAX_LEN      = 200       # max tokens per review (longer = truncated)
EMBED_DIM    = 100       # embedding vector size
BATCH_SIZE   = 256
EPOCHS       = 6         # EarlyStopping usually stops before this

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================================================================== #
#  1. LOAD the balanced sample saved by Stage 1
# ================================================================== #
# parquet may be a directory of part-files (Spark) or a single file;
# pandas + pyarrow reads both.
df = pd.read_parquet(PARQUET_PATH)
df = df[["clean_text", "label"]].dropna()
df["clean_text"] = df["clean_text"].astype(str)
df["label"] = df["label"].astype(int)

print(f"[LOAD] rows: {len(df):,}  "
      f"(pos={int((df.label==1).sum()):,}, neg={int((df.label==0).sum()):,})")

# ================================================================== #
#  2. RECREATE the disjoint splits, matching Stage 1's philosophy
# ------------------------------------------------------------------ #
#  - TRAIN + BALANCED-TEST come from the balanced sample (80/20).
#  - REALISTIC-TEST is rebuilt at the TRUE class ratio from the rows
#    NOT used in training, so metrics reflect real-world imbalance.
#  We reproduce the ~78% positive prior reported in Stage 1.
# ================================================================== #
REAL_POS_RATIO = 0.78   # matches the dataset's real balance (Stage 1 prints the exact value)

train_df, test_bal = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["label"]
)

# realistic test: sample from the held-out balanced-test pool at the
# true ratio (kept simple + disjoint from train by construction)
pos_pool = test_bal[test_bal.label == 1]
neg_pool = test_bal[test_bal.label == 0]
n_real   = min(len(pos_pool) + len(neg_pool), 8000)
n_pos    = int(n_real * REAL_POS_RATIO)
n_neg    = n_real - n_pos
n_pos    = min(n_pos, len(pos_pool))
n_neg    = min(n_neg, len(neg_pool))
test_real = pd.concat([
    pos_pool.sample(n_pos, random_state=SEED),
    neg_pool.sample(n_neg, random_state=SEED),
]).sample(frac=1, random_state=SEED)   # shuffle

print(f"[SPLIT] train={len(train_df):,}  "
      f"balanced-test={len(test_bal):,}  "
      f"realistic-test={len(test_real):,}")

# ================================================================== #
#  3. TEXT -> INTEGER SEQUENCES  (fit tokenizer on TRAIN ONLY)
# ------------------------------------------------------------------ #
#  Fitting the tokenizer only on training text prevents information
#  from the test set leaking into the vocabulary.
# ================================================================== #
tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tok.fit_on_texts(train_df["clean_text"])

def to_padded(texts):
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

X_train = to_padded(train_df["clean_text"]); y_train = train_df["label"].values
X_bal   = to_padded(test_bal["clean_text"]); y_bal   = test_bal["label"].values
X_real  = to_padded(test_real["clean_text"]); y_real = test_real["label"].values

vocab_size = min(MAX_WORDS, len(tok.word_index) + 1)
print(f"[VOCAB] vocabulary size: {vocab_size:,}")

# ================================================================== #
#  4. BUILD the LSTM
# ------------------------------------------------------------------ #
#  A compact, well-regularized Bi-LSTM. Bidirectional lets the model
#  read context from both directions, which helps with negation
#  ("not fresh", "never buying again") — a known weak spot for
#  bag-of-words models and a nice point to raise in your report.
# ================================================================== #
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.0)),
    Dense(32, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid"),   # binary output
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# ================================================================== #
#  5. TRAIN
# ================================================================== #
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early],
    verbose=2,
)

# ================================================================== #
#  6. EVALUATE  (same metric shape as Stage 1)
# ================================================================== #
def report(name, X, y, tag):
    prob = model.predict(X, batch_size=BATCH_SIZE, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    acc  = accuracy_score(y, pred)
    prec = precision_score(y, pred, average="weighted", zero_division=0)
    rec  = recall_score(y, pred, average="weighted", zero_division=0)
    f1   = f1_score(y, pred, average="weighted", zero_division=0)
    print(f"  [{tag}] {name:<12} acc={acc:.3f}  prec={prec:.3f}  "
          f"rec={rec:.3f}  f1={f1:.3f}")
    return dict(accuracy=acc, weightedPrecision=prec,
                weightedRecall=rec, f1=f1)

print("\n[EVAL]")
s_bal  = report("Bi-LSTM", X_bal,  y_bal,  "balanced ")
s_real = report("Bi-LSTM", X_real, y_real, "realistic")

# detailed per-class view on the realistic test (for the report)
pred_real = (model.predict(X_real, batch_size=BATCH_SIZE, verbose=0).ravel() >= 0.5).astype(int)
print("\n[REALISTIC TEST] per-class report")
print(classification_report(y_real, pred_real,
                            target_names=["negative", "positive"],
                            zero_division=0))
print("[REALISTIC TEST] confusion matrix  [[TN FP] [FN TP]]")
print(confusion_matrix(y_real, pred_real))

# ================================================================== #
#  7. ONE-LINE SUMMARY for the shared comparison table
# ------------------------------------------------------------------ #
#  Paste this row alongside the classical-model rows from Stage 1.
# ================================================================== #
print("\n" + "=" * 68)
print("Bi-LSTM  (realistic / imbalanced test set)")
print("=" * 68)
print(f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
print("-" * 68)
print(f"{'Bi-LSTM':<20}"
      f"{s_real['accuracy']:>10.3f}"
      f"{s_real['weightedPrecision']:>11.3f}"
      f"{s_real['weightedRecall']:>9.3f}"
      f"{s_real['f1']:>8.3f}")
print("=" * 68)
