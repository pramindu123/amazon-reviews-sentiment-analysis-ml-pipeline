"""
=====================================================================
 STAGE 1 — ML BACKBONE
 Machine Learning-Based Sentiment Analysis of Amazon Fine Food Reviews
 Module: ITC 4378 - Big Data Analytics
---------------------------------------------------------------------
 This stage does four things:
   1. Load the raw Amazon Fine Food Reviews (PySpark).
   2. Preprocess: de-duplicate, clean, build a binary sentiment label.
   3. Sample correctly: STRATIFIED sampling with sampleBy() to break
      the time-ordering bias and control class balance. We build a
      BALANCED training set but keep an IMBALANCED (realistic) test
      set so our reported metrics are honest.
   4. Train + compare classical ML models on TF-IDF features
      (Logistic Regression, Linear SVM, Naive Bayes) and report
      accuracy, precision, recall and F1.
=====================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
)
from pyspark.ml.classification import (
    LogisticRegression, LinearSVC, NaiveBayes
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# ------------------------------------------------------------------ #
#  CONFIGURATION  (change these two lines and nothing else)
# ------------------------------------------------------------------ #
DATA_PATH   = "Reviews.csv"      # path to the Kaggle Reviews.csv
SAMPLE_SIZE = 150_000            # target size of the BALANCED sample
SEED        = 42                 # fixed seed => reproducible results

# ================================================================== #
#  0. SPARK SESSION
# ================================================================== #
spark = (
    SparkSession.builder
    .appName("AmazonFineFood-Stage1")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# ================================================================== #
#  1. LOAD
# ================================================================== #
# multiLine + escape handle review text that contains commas / quotes
raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv(DATA_PATH)
)

print(f"[LOAD] raw rows: {raw.count():,}")

# ================================================================== #
#  2. PREPROCESS
# ================================================================== #

# 2a. Drop rows with no review text or no score
df = raw.dropna(subset=["Text", "Score"])

# 2b. Remove logically-invalid helpfulness rows
#     (a review can't be found helpful by MORE people than voted on it)
df = df.filter(F.col("HelpfulnessNumerator") <= F.col("HelpfulnessDenominator"))

# 2c. Remove exact-duplicate reviews.
#     The dataset is known to repeat the same review across ProductIds.
#     Same user + same text + same timestamp = the same review.
before = df.count()
df = df.dropDuplicates(["UserId", "Time", "Text"])
print(f"[CLEAN] removed {before - df.count():,} duplicate reviews")

# 2d. Build the binary sentiment label from Score.
#     Score 4-5  -> positive (1)   |   Score 1-2 -> negative (0)
#     Score 3    -> neutral, DROPPED for binary classification.
df = df.withColumn(
    "label",
    F.when(F.col("Score") >= 4, 1)     # positive
     .when(F.col("Score") <= 2, 0)     # negative
     .otherwise(None)                  # neutral -> drop
).dropna(subset=["label"])
df = df.withColumn("label", F.col("label").cast("int"))

# 2e. Light text cleaning: lower-case, strip HTML tags and non-letters.
df = df.withColumn("clean_text", F.lower(F.col("Text")))
df = df.withColumn("clean_text", F.regexp_replace("clean_text", r"<[^>]+>", " "))     # html
df = df.withColumn("clean_text", F.regexp_replace("clean_text", r"[^a-z\s]", " "))    # keep letters
df = df.withColumn("clean_text", F.regexp_replace("clean_text", r"\s+", " "))         # collapse spaces
df = df.filter(F.length("clean_text") > 3)

# keep only the columns we need downstream (Helpfulness kept for Stage 3)
df = df.select(
    "clean_text", "label", "Score",
    "HelpfulnessNumerator", "HelpfulnessDenominator", "ProductId", "Time"
).cache()

total = df.count()
pos   = df.filter(F.col("label") == 1).count()
neg   = df.filter(F.col("label") == 0).count()
print(f"[LABEL] usable reviews: {total:,}  (positive={pos:,}, negative={neg:,})")
print(f"[LABEL] real-world balance: {pos/total:.1%} positive / {neg/total:.1%} negative")

# ================================================================== #
#  3. STRATIFIED SAMPLING  ("sampling with ordered data")
# ------------------------------------------------------------------ #
#  WHY: the raw data is ordered by time and clustered by product, and
#  is ~78% positive. A naive .sample() would keep that ordering bias
#  and that imbalance. sampleBy() draws a chosen FRACTION FROM EACH
#  CLASS, which (a) breaks the ordering and (b) lets us fix balance.
#
#  We want a BALANCED sample of ~SAMPLE_SIZE, so half from each class.
#  If a class is too small to supply its half, we cap at that class.
# ================================================================== #
from pyspark.sql.functions import monotonically_increasing_id

# Give every cleaned review a stable unique id FIRST, and materialize
# it, so the same ids are used by both the sample and the later
# anti-join that guarantees train/test sets never overlap.
df = df.withColumn("row_id", monotonically_increasing_id()).cache()
df.count()  # force evaluation -> ids are now fixed

# fraction to draw from each class to hit a balanced ~SAMPLE_SIZE.
# If a class is too small to supply its half, cap at that class.
per_class = SAMPLE_SIZE // 2
frac_pos  = min(1.0, per_class / pos)
frac_neg  = min(1.0, per_class / neg)

sample = df.sampleBy("label", fractions={1: frac_pos, 0: frac_neg}, seed=SEED)
# Shuffle AFTER sampling so any residual sequence structure is destroyed
sample = sample.orderBy(F.rand(SEED)).cache()

s_total = sample.count()
s_pos   = sample.filter(F.col("label") == 1).count()
s_neg   = sample.filter(F.col("label") == 0).count()
print(f"[SAMPLE] balanced sample size: {s_total:,}  (pos={s_pos:,}, neg={s_neg:,})")

# ------------------------------------------------------------------ #
#  TRAIN / TEST SPLIT  (all three sets guaranteed disjoint)
#  - TRAIN on the BALANCED sample (model learns both classes equally).
#  - BALANCED TEST: held-out 20% of the balanced sample -> fair,
#    per-class view of performance.
#  - REALISTIC TEST: drawn from cleaned rows NOT used in training,
#    at the TRUE class imbalance -> honest real-world metrics.
# ------------------------------------------------------------------ #
train_df, test_bal = sample.randomSplit([0.8, 0.2], seed=SEED)
train_df = train_df.cache()
test_bal = test_bal.cache()

# rows available for the realistic test = everything NOT in training
train_ids = train_df.select("row_id")
pool      = df.join(train_ids, on="row_id", how="left_anti")   # anti-join

# draw ~15k from the pool at the TRUE positive/negative ratio
real_ratio_pos = pos / total
test_real = pool.sampleBy(
    "label",
    fractions={1: min(1.0, 15000 * real_ratio_pos / pos),
               0: min(1.0, 15000 * (1 - real_ratio_pos) / neg)},
    seed=SEED + 1
).cache()

print(f"[SPLIT] train={train_df.count():,}  "
      f"balanced-test={test_bal.count():,}  "
      f"realistic-test={test_real.count():,}  (all disjoint)")

# ================================================================== #
#  4. FEATURE PIPELINE  (TF-IDF)
# ================================================================== #
tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words",
                           pattern="\\s+", minTokenLength=2)
remover   = StopWordsRemover(inputCol="words", outputCol="filtered")

# HashingTF is fast and memory-stable at scale; IDF re-weights terms.
hashing   = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=1 << 16)
idf       = IDF(inputCol="tf", outputCol="features")

feature_stages = [tokenizer, remover, hashing, idf]

# ================================================================== #
#  5. MODELS + EVALUATION
# ================================================================== #
models = {
    "LogisticRegression": LogisticRegression(maxIter=50, regParam=0.01),
    "LinearSVM":          LinearSVC(maxIter=50, regParam=0.01),
    "NaiveBayes":         NaiveBayes(modelType="multinomial"),  # needs TF, see note
}

# metrics we report
metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]
evaluators = {
    m: MulticlassClassificationEvaluator(labelCol="label",
                                         predictionCol="prediction",
                                         metricName=m)
    for m in metrics
}

def evaluate(name, model, test_df, tag):
    preds = model.transform(test_df)
    scores = {m: evaluators[m].evaluate(preds) for m in metrics}
    print(f"  [{tag}] {name:<20} "
          f"acc={scores['accuracy']:.3f}  "
          f"prec={scores['weightedPrecision']:.3f}  "
          f"rec={scores['weightedRecall']:.3f}  "
          f"f1={scores['f1']:.3f}")
    return scores

results = {}
for name, clf in models.items():
    print(f"\n[TRAIN] {name}")

    # Naive Bayes multinomial needs raw counts (non-negative), so it
    # uses TF directly rather than IDF-weighted features.
    if name == "NaiveBayes":
        stages = [tokenizer, remover, hashing]
        clf.setFeaturesCol("tf")
    else:
        stages = feature_stages
        clf.setFeaturesCol("features")

    pipe  = Pipeline(stages=stages + [clf])
    model = pipe.fit(train_df)

    # report BOTH: balanced test (fair per-class view) and realistic test
    s_bal  = evaluate(name, model, test_bal,  "balanced ")
    s_real = evaluate(name, model, test_real, "realistic")
    results[name] = {"balanced": s_bal, "realistic": s_real}

# ================================================================== #
#  6. SUMMARY TABLE
# ================================================================== #
print("\n" + "=" * 68)
print("STAGE 1 RESULTS  (realistic / imbalanced test set)")
print("=" * 68)
print(f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
print("-" * 68)
for name, r in results.items():
    s = r["realistic"]
    print(f"{name:<20}"
          f"{s['accuracy']:>10.3f}"
          f"{s['weightedPrecision']:>11.3f}"
          f"{s['weightedRecall']:>9.3f}"
          f"{s['f1']:>8.3f}")
print("=" * 68)

# Save the balanced sample for Stages 2 & 3 (aspect layer + insights)
(sample.select("clean_text", "label", "Score",
               "HelpfulnessNumerator", "HelpfulnessDenominator",
               "ProductId", "Time")
       .write.mode("overwrite").parquet("sample_stage1.parquet"))
print("\n[SAVE] balanced sample written to sample_stage1.parquet "
      "(input for Stage 2)")

spark.stop()
