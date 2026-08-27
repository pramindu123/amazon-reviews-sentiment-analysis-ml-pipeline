# Sentiment Analysis ML Pipeline

Local VS Code project for an AWS-based sentiment analysis pipeline, intended for development with the AWS Toolkit and collaboration through GitHub.

## Architecture

Raw data lands at `s3://<bucket>/raw/`. AWS Glue PySpark jobs process it through bronze (minimal cleanup and ingestion metadata), silver (lowercasing, HTML and punctuation removal, deduplication, and null removal), and gold (vectorized features and labels), writing each layer to its corresponding S3 prefix.

AWS SageMaker reads `s3://<bucket>/gold/` and trains a TF-IDF plus Logistic Regression classifier with a scikit-learn script-mode Estimator. Model artifacts are saved under `s3://<bucket>/models/`.
