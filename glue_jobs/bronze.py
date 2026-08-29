from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


spark = SparkSession.builder.appName("amazon-reviews-raw-to-bronze").getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

RAW_PATH = "s3://amazon-food-reviews-ml-model/raw/amazon.csv"
BRONZE_PATH = "s3://amazon-food-reviews-ml-model/bronze/"

EXPECTED_COLUMNS = [
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

# Read every raw field as text first so Bronze controls every type conversion.
raw_schema = StructType(
    [StructField(column, StringType(), True) for column in EXPECTED_COLUMNS]
)

raw_df = (
    spark.read
    .schema(raw_schema)
    .option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    # Keep structurally readable rows and let the validation rules below
    # reject bad values without terminating the entire Glue job.
    .option("mode", "PERMISSIVE")
    .csv(RAW_PATH)
)

if raw_df.columns != EXPECTED_COLUMNS:
    raise ValueError(
        f"Unexpected raw schema. Expected {EXPECTED_COLUMNS}, got {raw_df.columns}"
    )

def safe_integral(column_name, data_type):
    """Cast an integer-like string without failing under Spark ANSI mode."""
    value = F.trim(F.col(column_name))
    return F.when(value.rlike(r"^[+-]?[0-9]+$"), value.cast(data_type))


source_file = F.input_file_name()
time_raw = F.trim(F.col("Time"))
review_time_epoch = safe_integral("Time", "long")

bronze_df = (
    raw_df
    .select(
        safe_integral("Id", "long").alias("id"),
        F.trim(F.col("ProductId")).alias("product_id"),
        F.trim(F.col("UserId")).alias("user_id"),
        F.trim(F.col("ProfileName")).alias("profile_name"),
        safe_integral("HelpfulnessNumerator", "int").alias("helpfulness_numerator"),
        safe_integral("HelpfulnessDenominator", "int").alias("helpfulness_denominator"),
        safe_integral("Score", "int").alias("score"),
        time_raw.alias("review_time_raw"),
        review_time_epoch.alias("review_time_epoch"),
        F.from_unixtime(review_time_epoch).cast("timestamp").alias("review_timestamp"),
        F.trim(F.col("Summary")).alias("summary"),
        F.trim(F.col("Text")).alias("review_text"),
        source_file.alias("_source_file"),
    )
    # Time is deliberately not required. The sample's spreadsheet-exported
    # values are invalid, but its review text and scores remain usable.
    .withColumn(
        "is_valid_record",
        F.col("id").isNotNull()
        & F.col("score").between(1, 5)
        & F.col("review_text").isNotNull()
        & (F.length(F.col("review_text")) > 0)
        & F.col("helpfulness_numerator").isNotNull()
        & F.col("helpfulness_denominator").isNotNull()
        & (F.col("helpfulness_numerator") <= F.col("helpfulness_denominator")),
    )
    .filter(F.col("is_valid_record"))
    .withColumn("positive", F.when(F.col("score") >= 4, 1).otherwise(0).cast("int"))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn(
        "_record_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("_source_file"), F.lit("")),
                F.coalesce(F.col("id").cast("string"), F.lit("")),
                F.coalesce(F.col("product_id"), F.lit("")),
                F.coalesce(F.col("user_id"), F.lit("")),
            ),
            256,
        ),
    )
)

bronze_count = bronze_df.count()
if bronze_count == 0:
    raise ValueError("Bronze transformation produced zero valid records")

print(f"Valid Bronze rows: {bronze_count}")
bronze_df.groupBy("score", "positive").count().orderBy("score").show()
bronze_df.select(
    F.count(F.when(F.col("review_time_epoch").isNull(), 1)).alias(
        "rows_with_invalid_time"
    )
).show()

(
    bronze_df.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(BRONZE_PATH)
)

# Read the output back so a failed or incomplete write is detected immediately.
written_count = spark.read.parquet(BRONZE_PATH).count()
if written_count != bronze_count:
    raise ValueError(
        f"Bronze write verification failed: expected {bronze_count}, got {written_count}"
    )

print(f"Saved {written_count} rows as Parquet to {BRONZE_PATH}")
