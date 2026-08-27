from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

silver_path = "s3://amazon-food-reviews-ml-model/silver/"
gold_path = "s3://amazon-food-reviews-ml-model/gold/"

df = spark.read.parquet(silver_path)

gold_df = df.select(
    col("_record_id").alias("record_id"),
    "clean_text",
    col("positive").alias("label"),
    "review_length"
)

gold_df.write.mode("overwrite").parquet(gold_path)