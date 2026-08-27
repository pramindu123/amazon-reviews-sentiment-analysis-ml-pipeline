from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, regexp_replace, length

spark = SparkSession.builder.getOrCreate()

bronze_path = "s3://amazon-food-reviews-ml-model/bronze/"
silver_path = "s3://amazon-food-reviews-ml-model/silver/"

df = spark.read.parquet(bronze_path)

silver_df = (
    df
    .filter(col("review_text").isNotNull())
    .filter(col("positive").isin(0, 1))
    .dropDuplicates(["review_text"])
    .withColumn("review_text", trim(col("review_text")))
    .withColumn("clean_text", lower(col("review_text")))
    .withColumn(
        "clean_text",
        regexp_replace(col("clean_text"), r"[^a-z0-9\s]", " ")
    )
    .withColumn(
        "clean_text",
        regexp_replace(col("clean_text"), r"\s+", " ")
    )
    .withColumn("clean_text", trim(col("clean_text")))
    .withColumn("review_length", length(col("review_text")))
    .filter(length(col("clean_text")) > 0)
)

silver_df.write.mode("overwrite").parquet(silver_path)