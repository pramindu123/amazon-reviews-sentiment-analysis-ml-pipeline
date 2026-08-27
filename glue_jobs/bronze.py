from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    input_file_name,
    monotonically_increasing_id
)

spark = SparkSession.builder.getOrCreate()

raw_path = "s3://amazon-food-reviews-ml-model/raw/amazon.csv"
bronze_path = "s3://amazon-food-reviews-ml-model/bronze/"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(raw_path)
)

bronze_df = (
    df
    .withColumnRenamed("reviewText", "review_text")
    .withColumnRenamed("Positive", "positive")
    .withColumn("positive", col("positive").cast("int"))
    .withColumn("_ingested_at", current_timestamp())
    .withColumn("_source_file", input_file_name())
    .withColumn("_record_id", monotonically_increasing_id())
)

bronze_df.write.mode("overwrite").parquet(bronze_path)