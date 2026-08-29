"""Ingest the Amazon Fine Food Reviews CSV into the Bronze layer.

Bronze is intentionally lossless: source values remain strings, no ratings are
relabelled, and no records are discarded. Typing, validation and deduplication
belong to Silver.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


spark = SparkSession.builder.appName("amazon-reviews-raw-to-bronze").getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

RAW_PATH = "s3://amazon-food-reviews-ml-model/raw/Reviews.csv"
BRONZE_PATH = "s3://amazon-food-reviews-ml-model/bronze/"

RAW_COLUMNS = [
    "Id",
    "ProductId",
    "UserId",
    "ProfileName",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
    "Score",
    "Time",
    "Summary",
    "Text",
]

raw_schema = StructType(
    [StructField(column_name, StringType(), True) for column_name in RAW_COLUMNS]
)

raw_df = (
    spark.read
    .schema(raw_schema)
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("mode", "PERMISSIVE")
    .csv(RAW_PATH)
)

if raw_df.columns != RAW_COLUMNS:
    raise ValueError(
        f"Unexpected raw schema. Expected {RAW_COLUMNS}, got {raw_df.columns}"
    )

bronze_df = (
    raw_df.select(
        F.col("Id").alias("raw_id"),
        F.col("ProductId").alias("product_id"),
        F.col("UserId").alias("user_id"),
        F.col("ProfileName").alias("profile_name"),
        F.col("HelpfulnessNumerator").alias("raw_helpfulness_numerator"),
        F.col("HelpfulnessDenominator").alias("raw_helpfulness_denominator"),
        F.col("Score").alias("raw_score"),
        F.col("Time").alias("raw_review_time"),
        F.col("Summary").alias("summary"),
        F.col("Text").alias("review_text"),
        F.input_file_name().alias("_source_file"),
    )
    .withColumn("_bronze_ingested_at", F.current_timestamp())
    .withColumn(
        "_record_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("_source_file"), F.lit("")),
                F.coalesce(F.col("raw_id"), F.lit("")),
                F.coalesce(F.col("product_id"), F.lit("")),
                F.coalesce(F.col("user_id"), F.lit("")),
                F.coalesce(F.col("raw_review_time"), F.lit("")),
            ),
            256,
        ),
    )
)

raw_count = raw_df.count()
bronze_count = bronze_df.count()
if raw_count == 0:
    raise ValueError(f"No raw records found at {RAW_PATH}")
if bronze_count != raw_count:
    raise ValueError(
        f"Bronze must be lossless: raw={raw_count}, bronze={bronze_count}"
    )

(
    bronze_df.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(BRONZE_PATH)
)

written_count = spark.read.parquet(BRONZE_PATH).count()
if written_count != bronze_count:
    raise ValueError(
        f"Bronze write verification failed: expected {bronze_count}, got {written_count}"
    )

print(f"Raw rows: {raw_count:,}")
print(f"Saved {written_count:,} lossless Bronze rows to {BRONZE_PATH}")
