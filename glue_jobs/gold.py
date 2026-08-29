"""Publish ML-ready and aspect-ready Gold data products.

No model is trained here. Gold prepares two independent contracts:

* model_input: deduplicated binary sentiment rows with deterministic,
  label-stratified train/validation/test assignments.
* aspect_sentences: punctuation-preserving sentence rows with product, time and
  helpfulness context for future lexicon/LDA/aspect-sentiment processing.

Class balancing must be applied only to the training split during model
development. Validation and test data retain the observed class distribution.
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


spark = SparkSession.builder.appName("amazon-reviews-silver-to-gold").getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

SILVER_PATH = "s3://amazon-food-reviews-ml-model/silver/"
GOLD_MODEL_PATH = "s3://amazon-food-reviews-ml-model/gold/model_input/"
GOLD_ASPECT_PATH = "s3://amazon-food-reviews-ml-model/gold/aspect_sentences/"

REQUIRED_COLUMNS = {
    "id",
    "product_id",
    "user_id",
    "score",
    "sentiment_class",
    "binary_label",
    "review_text_clean",
    "normalized_text",
    "review_length",
    "review_word_count",
    "review_time_epoch",
    "review_timestamp",
    "helpfulness_numerator",
    "helpfulness_denominator",
    "helpfulness_ratio",
    "_record_id",
}

silver_df = spark.read.parquet(SILVER_PATH)
missing_columns = sorted(REQUIRED_COLUMNS - set(silver_df.columns))
if missing_columns:
    raise ValueError(f"Silver data is missing required columns: {missing_columns}")

# ---------------------------------------------------------------------------
# Gold product 1: binary sentiment model input
# ---------------------------------------------------------------------------
# Keep the most-supported copy of identical normalized text. This prevents the
# same review text leaking across future train and evaluation splits.
model_duplicate_window = Window.partitionBy("normalized_text").orderBy(
    F.col("helpfulness_denominator").desc(),
    F.col("id").asc(),
    F.col("_record_id").asc(),
)

deduplicated_model_df = (
    silver_df
    .filter(F.col("binary_label").isin(0, 1))
    .withColumn("_model_duplicate_rank", F.row_number().over(model_duplicate_window))
    .filter(F.col("_model_duplicate_rank") == 1)
    .drop("_model_duplicate_rank")
)

# Exact label-stratified 80/10/10 assignment using a deterministic hash order.
# Balancing is intentionally deferred until a future trainer samples TRAIN only.
split_order_window = Window.partitionBy("binary_label").orderBy(
    F.xxhash64("_record_id"), F.col("_record_id")
)
label_window = Window.partitionBy("binary_label")

model_input_df = (
    deduplicated_model_df
    .withColumn("_label_row_number", F.row_number().over(split_order_window))
    .withColumn("_label_count", F.count(F.lit(1)).over(label_window))
    .withColumn(
        "_label_fraction",
        F.col("_label_row_number").cast("double") / F.col("_label_count"),
    )
    .withColumn(
        "dataset_split",
        F.when(F.col("_label_fraction") <= 0.80, F.lit("train"))
        .when(F.col("_label_fraction") <= 0.90, F.lit("validation"))
        .otherwise(F.lit("test")),
    )
    .select(
        F.col("_record_id").alias("record_id"),
        F.col("normalized_text").alias("clean_text"),
        F.col("binary_label").cast("int").alias("label"),
        "dataset_split",
        "score",
        "product_id",
        "user_id",
        "review_timestamp",
        "review_length",
        "review_word_count",
        "helpfulness_numerator",
        "helpfulness_denominator",
        "helpfulness_ratio",
    )
)

# ---------------------------------------------------------------------------
# Gold product 2: sentence-level input for future food-domain ABSA
# ---------------------------------------------------------------------------
# Silver preserved punctuation, so sentence extraction remains possible. No
# lexicon, LDA, VADER or trained model is applied at this transformation stage.
sentence_ready_df = silver_df.withColumn(
    "_sentence_array",
    F.split(F.col("review_text_clean"), r"(?<=[.!?])\s+"),
)

aspect_sentences_df = (
    sentence_ready_df
    .select(
        "_record_id",
        "id",
        "product_id",
        "user_id",
        "score",
        "sentiment_class",
        "review_time_epoch",
        "review_timestamp",
        "helpfulness_numerator",
        "helpfulness_denominator",
        "helpfulness_ratio",
        F.posexplode("_sentence_array").alias("sentence_index", "sentence_text"),
    )
    .withColumn("sentence_text", F.trim(F.col("sentence_text")))
    .filter(F.length(F.col("sentence_text")) > 0)
    .withColumn(
        "sentence_normalized",
        F.trim(F.regexp_replace(F.lower(F.col("sentence_text")), r"\s+", " ")),
    )
    .withColumn("review_year", F.year(F.col("review_timestamp")))
    .withColumn(
        "sentence_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.col("_record_id"),
                F.col("sentence_index").cast("string"),
            ),
            256,
        ),
    )
    .select(
        "sentence_id",
        F.col("_record_id").alias("record_id"),
        "id",
        "product_id",
        "user_id",
        "score",
        "sentiment_class",
        "review_time_epoch",
        "review_timestamp",
        "review_year",
        "helpfulness_numerator",
        "helpfulness_denominator",
        "helpfulness_ratio",
        "sentence_index",
        "sentence_text",
        "sentence_normalized",
    )
)

model_count = model_input_df.count()
sentence_count = aspect_sentences_df.count()
if model_count == 0:
    raise ValueError("Gold model_input transformation produced zero rows")
if sentence_count == 0:
    raise ValueError("Gold aspect_sentences transformation produced zero rows")

(
    model_input_df.write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("dataset_split")
    .parquet(GOLD_MODEL_PATH)
)

(
    aspect_sentences_df.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(GOLD_ASPECT_PATH)
)

written_model_count = spark.read.parquet(GOLD_MODEL_PATH).count()
written_sentence_count = spark.read.parquet(GOLD_ASPECT_PATH).count()
if written_model_count != model_count:
    raise ValueError(
        f"Gold model write failed: expected {model_count}, got {written_model_count}"
    )
if written_sentence_count != sentence_count:
    raise ValueError(
        "Gold aspect write failed: "
        f"expected {sentence_count}, got {written_sentence_count}"
    )

print(f"Gold model_input rows: {written_model_count:,}")
model_input_df.groupBy("dataset_split", "label").count().orderBy(
    "dataset_split", "label"
).show()
print(f"Gold aspect_sentences rows: {written_sentence_count:,}")
print(f"Model output: {GOLD_MODEL_PATH}")
print(f"Aspect output: {GOLD_ASPECT_PATH}")
