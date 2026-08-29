"""Validate, type, clean and deduplicate Bronze Amazon reviews.

Silver preserves punctuation and sentence boundaries for future aspect-based
sentiment analysis. It assigns three rating classes, but does not incorrectly
treat 3-star reviews as negative. The nullable binary label is prepared only
as a convenience; Gold decides which rows belong in the binary ML dataset.
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


spark = SparkSession.builder.appName("amazon-reviews-bronze-to-silver").getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

BRONZE_PATH = "s3://amazon-food-reviews-ml-model/bronze/"
SILVER_PATH = "s3://amazon-food-reviews-ml-model/silver/"

REQUIRED_COLUMNS = {
    "raw_id",
    "product_id",
    "user_id",
    "profile_name",
    "raw_helpfulness_numerator",
    "raw_helpfulness_denominator",
    "raw_score",
    "raw_review_time",
    "summary",
    "review_text",
    "_source_file",
    "_record_id",
    "_bronze_ingested_at",
}


def safe_integral(column_name, data_type):
    """Cast an integer-like string without failing under Spark ANSI mode."""
    value = F.trim(F.col(column_name))
    return F.when(value.rlike(r"^[+-]?[0-9]+$"), value.cast(data_type))


bronze_df = spark.read.parquet(BRONZE_PATH)
missing_columns = sorted(REQUIRED_COLUMNS - set(bronze_df.columns))
if missing_columns:
    raise ValueError(f"Bronze data is missing required columns: {missing_columns}")

review_time_epoch = safe_integral("raw_review_time", "long")

typed_df = bronze_df.select(
    safe_integral("raw_id", "long").alias("id"),
    F.trim(F.col("product_id")).alias("product_id"),
    F.trim(F.col("user_id")).alias("user_id"),
    F.trim(F.col("profile_name")).alias("profile_name"),
    safe_integral("raw_helpfulness_numerator", "int").alias(
        "helpfulness_numerator"
    ),
    safe_integral("raw_helpfulness_denominator", "int").alias(
        "helpfulness_denominator"
    ),
    safe_integral("raw_score", "int").alias("score"),
    F.col("raw_review_time").alias("review_time_raw"),
    review_time_epoch.alias("review_time_epoch"),
    F.from_unixtime(review_time_epoch).cast("timestamp").alias("review_timestamp"),
    F.col("summary"),
    F.col("review_text"),
    F.col("_source_file"),
    F.col("_record_id"),
    F.col("_bronze_ingested_at"),
)

# Remove markup while retaining punctuation needed for sentence-level ABSA.
cleaned_df = (
    typed_df
    .withColumn("summary", F.trim(F.col("summary")))
    .withColumn("review_text_clean", F.col("review_text"))
    .withColumn(
        "review_text_clean",
        F.regexp_replace("review_text_clean", r"(?i)<br\s*/?>", ". "),
    )
    .withColumn(
        "review_text_clean",
        F.regexp_replace("review_text_clean", r"(?i)</?p\s*>", ". "),
    )
    .withColumn(
        "review_text_clean",
        F.regexp_replace("review_text_clean", r"(?i)<[^>]+>", " "),
    )
    .withColumn(
        "review_text_clean",
        F.regexp_replace("review_text_clean", r"(?i)&amp;", " and "),
    )
    .withColumn(
        "review_text_clean",
        F.regexp_replace("review_text_clean", r"(?i)&quot;", '"'),
    )
    .withColumn(
        "review_text_clean",
        F.regexp_replace("review_text_clean", r"(?i)&(?:apos|#39);", "'"),
    )
    .withColumn(
        "review_text_clean",
        F.regexp_replace(
            "review_text_clean", r"(?i)&(?:nbsp|lt|gt);|&#[0-9]+;", " "
        ),
    )
    .withColumn(
        "review_text_clean",
        F.trim(F.regexp_replace("review_text_clean", r"\s+", " ")),
    )
    .withColumn("normalized_text", F.lower(F.col("review_text_clean")))
    .withColumn(
        "normalized_text",
        F.trim(F.regexp_replace("normalized_text", r"\s+", " ")),
    )
)

valid_df = (
    cleaned_df
    .filter(F.col("id").isNotNull())
    .filter(F.col("product_id").isNotNull() & (F.length("product_id") > 0))
    .filter(F.col("user_id").isNotNull() & (F.length("user_id") > 0))
    .filter(F.col("score").between(1, 5))
    .filter(F.col("review_text_clean").isNotNull())
    .filter(F.length(F.col("review_text_clean")) > 0)
    .filter(F.col("helpfulness_numerator").isNotNull())
    .filter(F.col("helpfulness_denominator").isNotNull())
    .filter(F.col("helpfulness_numerator") >= 0)
    .filter(F.col("helpfulness_denominator") >= 0)
    .filter(F.col("helpfulness_numerator") <= F.col("helpfulness_denominator"))
)

# Remove repeated rows within a product. Cross-product copies are retained in
# Silver for product analytics; Gold removes text leakage for the ML dataset.
product_duplicate_window = Window.partitionBy(
    "product_id",
    "user_id",
    "review_time_epoch",
    "normalized_text",
).orderBy(F.col("id").asc(), F.col("_record_id").asc())

silver_df = (
    valid_df
    .withColumn("_product_duplicate_rank", F.row_number().over(product_duplicate_window))
    .filter(F.col("_product_duplicate_rank") == 1)
    .drop("_product_duplicate_rank")
    .withColumn(
        "sentiment_class",
        F.when(F.col("score") <= 2, F.lit("negative"))
        .when(F.col("score") == 3, F.lit("neutral"))
        .otherwise(F.lit("positive")),
    )
    .withColumn(
        "binary_label",
        F.when(F.col("score") <= 2, F.lit(0))
        .when(F.col("score") >= 4, F.lit(1))
        .otherwise(F.lit(None).cast("int")),
    )
    .withColumn("review_length", F.length(F.col("review_text_clean")))
    .withColumn(
        "review_word_count",
        F.size(F.split(F.col("normalized_text"), r"\s+")),
    )
    .withColumn(
        "helpfulness_ratio",
        F.when(
            F.col("helpfulness_denominator") > 0,
            F.col("helpfulness_numerator").cast("double")
            / F.col("helpfulness_denominator"),
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn("has_valid_review_time", F.col("review_time_epoch").isNotNull())
    .withColumn("_silver_processed_at", F.current_timestamp())
)

bronze_count = bronze_df.count()
silver_count = silver_df.count()
if silver_count == 0:
    raise ValueError("Silver transformation produced zero valid records")

(
    silver_df.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(SILVER_PATH)
)

written_count = spark.read.parquet(SILVER_PATH).count()
if written_count != silver_count:
    raise ValueError(
        f"Silver write verification failed: expected {silver_count}, got {written_count}"
    )

print(f"Bronze rows: {bronze_count:,}")
print(f"Valid deduplicated Silver rows: {written_count:,}")
silver_df.groupBy("score", "sentiment_class", "binary_label").count().orderBy(
    "score"
).show()
