"""
=====================================================================
 STAGE 2 — FOOD-DOMAIN ASPECT LAYER  (the research contribution)
 Machine Learning-Based Sentiment Analysis of Amazon Fine Food Reviews
 Module: ITC 4378 - Big Data Analytics
---------------------------------------------------------------------
 Stage 1 tells you WHETHER a review is positive or negative.
 Stage 2 tells you WHAT it is positive or negative ABOUT — taste,
 freshness, packaging, price, delivery, or health. That "what" is
 what makes this project food-specific and actionable, and it is
 what covers the research gap in your report.

 Method (hybrid, and defensible with no aspect-labelled data):

   A. SEEDED LEXICON  — food aspects drawn from your literature
      (taste, quality, freshness, packaging, price, delivery, health).
      Each aspect has a set of keywords.

   B. LDA VALIDATION  — run topic modelling on the reviews and show
      that these aspects actually EMERGE from the data. This turns
      "we assumed these aspects" into "the data confirms these
      aspects", which is a real finding you can write up.

   C. PER-ASPECT SENTIMENT — for each review, find which aspects are
      mentioned, then score the SENTENCES that mention them with
      VADER (sentence-level, handles negation, no training needed).
      Result: every review gets a sentiment PER aspect, not one label.

 Output: aspect_sentiment.parquet  -> input for Stage 3 (insights).
=====================================================================
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ------------------------------------------------------------------ #
#  CONFIGURATION
# ------------------------------------------------------------------ #
PARQUET_IN  = "sample_stage1.parquet"       # from Stage 1
PARQUET_OUT = "aspect_sentiment.parquet"    # -> Stage 3
SEED        = 42
N_TOPICS    = 8      # LDA topics (~ number of seeded aspects)
POS_TH      = 0.05   # VADER compound >= this  -> positive
NEG_TH      = -0.05  # VADER compound <= this  -> negative

# ================================================================== #
#  A. SEEDED FOOD-ASPECT LEXICON
# ------------------------------------------------------------------ #
#  Keywords are intentionally food-domain. Extend these freely — the
#  LDA step (B) will suggest words you may have missed.
# ================================================================== #
ASPECTS = {
    "taste":     ["taste", "tastes", "tasty", "flavor", "flavour", "flavors",
                  "delicious", "yummy", "bland", "bitter", "sweet", "sour",
                  "savory", "aftertaste", "spicy"],
    "quality":   ["quality", "fresh", "freshness", "stale", "spoiled",
                  "rotten", "mold", "moldy", "expired", "authentic",
                  "genuine", "premium"],
    "freshness": ["fresh", "freshness", "stale", "old", "expired",
                  "expiry", "expiration", "date", "crisp", "soggy"],
    "packaging": ["packaging", "package", "packed", "packet", "box",
                  "sealed", "seal", "container", "wrapped", "leak",
                  "leaked", "leaking", "crushed", "damaged", "broken",
                  "bag", "bottle", "lid"],
    "price":     ["price", "priced", "expensive", "cheap", "cost",
                  "costly", "value", "worth", "overpriced", "affordable",
                  "money", "deal", "bargain"],
    "delivery":  ["delivery", "shipping", "shipped", "arrived", "arrival",
                  "melted", "late", "delayed", "fast", "quick", "prime",
                  "courier", "delivered"],
    "health":    ["healthy", "health", "organic", "natural", "sugar",
                  "sugarfree", "gluten", "calorie", "calories", "diet",
                  "ingredient", "ingredients", "preservative",
                  "preservatives", "additive", "allergy", "allergic",
                  "protein", "vegan"],
    "portion":   ["portion", "size", "quantity", "amount", "small",
                  "large", "big", "tiny", "generous", "count", "pack"],
}

# precompile a word-boundary regex per aspect for fast matching
ASPECT_RE = {
    a: re.compile(r"\b(" + "|".join(map(re.escape, kws)) + r")\b")
    for a, kws in ASPECTS.items()
}

# ================================================================== #
#  1. LOAD Stage 1 sample
# ------------------------------------------------------------------ #
#  We keep the metadata columns (helpfulness, product, time) so
#  Stage 3 can weight complaints by helpfulness and track them over
#  time. They are carried through untouched.
# ================================================================== #
raw = pd.read_parquet(PARQUET_IN)

KEEP_META = ["HelpfulnessNumerator", "HelpfulnessDenominator",
             "ProductId", "Time"]
present_meta = [c for c in KEEP_META if c in raw.columns]

df = raw[["clean_text", "label"] + present_meta].copy()
df["clean_text"] = df["clean_text"].astype(str)
if present_meta:
    print(f"[LOAD] carried metadata columns: {present_meta}")
else:
    print("[LOAD] no metadata columns found — Stage 3 will use "
          "unweighted counts only")
print(f"[LOAD] reviews: {len(df):,}")

# ================================================================== #
#  B. LDA VALIDATION  —  do the seeded aspects actually appear?
# ------------------------------------------------------------------ #
#  We fit LDA and print the top words per topic. In your report you
#  then MAP each topic to a seeded aspect ("topic 3 = packaging:
#  box, sealed, leaked, crushed"), and note any EMERGENT theme the
#  lexicon missed. That mapping is a genuine result.
# ================================================================== #
print("\n[LDA] fitting topic model to validate aspects ...")
vect = CountVectorizer(max_df=0.5, min_df=20, stop_words="english",
                       max_features=5000)
dtm = vect.fit_transform(df["clean_text"])
vocab = np.array(vect.get_feature_names_out())

lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=SEED,
                                learning_method="online", max_iter=15)
lda.fit(dtm)

print(f"[LDA] top words per topic (map these to your aspects):")
for k, comp in enumerate(lda.components_):
    top = vocab[comp.argsort()[-12:][::-1]]
    print(f"  topic {k}: {', '.join(top)}")

# ---- auto-suggest which seeded aspect each topic looks like -------- #
#  (a helper for your write-up; you still decide the final mapping)
print("\n[LDA] auto-suggested aspect per topic (heuristic):")
for k, comp in enumerate(lda.components_):
    top = set(vocab[comp.argsort()[-25:][::-1]])
    best_aspect, best_hits = None, 0
    for a, kws in ASPECTS.items():
        hits = len(top.intersection(kws))
        if hits > best_hits:
            best_aspect, best_hits = a, hits
    label = best_aspect if best_hits else "(emergent / unclear)"
    print(f"  topic {k} -> {label}  (keyword overlap: {best_hits})")

# ================================================================== #
#  C. PER-ASPECT SENTIMENT
# ------------------------------------------------------------------ #
#  For each review:
#    1. split into sentences
#    2. for each aspect, collect sentences that mention it
#    3. score those sentences with VADER -> avg compound
#    4. bucket into positive / neutral / negative
#  A review with no mention of an aspect gets NaN for that aspect.
# ================================================================== #
sia = SentimentIntensityAnalyzer()

def split_sentences(text):
    # lightweight splitter (clean_text has no punctuation after Stage 1
    # cleaning, so we also fall back to the whole text as one "sentence")
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p for p in parts if p.strip()]
    return parts if parts else [text]

def aspect_sentiments(text):
    sents = split_sentences(text)
    out = {}
    for aspect, rex in ASPECT_RE.items():
        hits = [s for s in sents if rex.search(s)]
        if not hits:
            out[aspect] = np.nan          # aspect not mentioned
        else:
            comp = np.mean([sia.polarity_scores(s)["compound"] for s in hits])
            out[aspect] = comp
    return out

print("\n[ASPECT] scoring per-aspect sentiment ...")
asp_df = df["clean_text"].apply(aspect_sentiments).apply(pd.Series)

# add compound columns to the frame
result = pd.concat([df.reset_index(drop=True),
                    asp_df.reset_index(drop=True)], axis=1)

# also add a categorical label per aspect (pos / neg / neu / not-mentioned)
def bucket(v):
    if pd.isna(v):        return "not_mentioned"
    if v >= POS_TH:       return "positive"
    if v <= NEG_TH:       return "negative"
    return "neutral"

for a in ASPECTS:
    result[f"{a}_sent"] = result[a].apply(bucket)

# ================================================================== #
#  2. QUICK SUMMARY  (a preview of the Stage-3 insights)
# ================================================================== #
print("\n" + "=" * 60)
print("ASPECT COVERAGE & SENTIMENT (share of all reviews)")
print("=" * 60)
print(f"{'Aspect':<12}{'mentioned':>10}{'%pos':>8}{'%neg':>8}")
print("-" * 60)
for a in ASPECTS:
    col = result[f"{a}_sent"]
    mentioned = (col != "not_mentioned").mean()
    among = col[col != "not_mentioned"]
    pos = (among == "positive").mean() if len(among) else 0
    neg = (among == "negative").mean() if len(among) else 0
    print(f"{a:<12}{mentioned:>9.1%}{pos:>8.0%}{neg:>8.0%}")
print("=" * 60)

# ================================================================== #
#  3. SAVE for Stage 3
# ================================================================== #
result.to_parquet(PARQUET_OUT, index=False)
print(f"\n[SAVE] per-aspect sentiment written to {PARQUET_OUT} "
      f"({len(result):,} rows)  -> input for Stage 3")
