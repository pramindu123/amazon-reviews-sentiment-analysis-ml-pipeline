# Sentiment Analysis ML Pipeline

Local VS Code project for an AWS-based sentiment analysis pipeline, intended for development with the AWS Toolkit and collaboration through GitHub.

## Medallion architecture

Raw data lands at `s3://amazon-food-reviews-ml-model/raw/Reviews.csv` and is transformed by AWS Glue PySpark jobs:

- **Bronze** (`s3://amazon-food-reviews-ml-model/bronze/`) is a lossless Parquet copy. All source values remain strings and receive ingestion metadata and a deterministic record ID. No review is relabelled or discarded.
- **Silver** (`s3://amazon-food-reviews-ml-model/silver/`) performs safe type conversion, data-quality filtering, HTML cleanup, whitespace normalization, product-safe deduplication, rating-class derivation, and helpfulness/text enrichment. Punctuation is retained for future sentence-level aspect analysis. Scores 1–2 are negative, 3 is neutral, and 4–5 are positive.
- **Gold model input** (`s3://amazon-food-reviews-ml-model/gold/model_input/`) contains globally deduplicated binary-labelled reviews and deterministic, label-stratified train/validation/test assignments. Neutral reviews are excluded only from this binary data product.
- **Gold aspect input** (`s3://amazon-food-reviews-ml-model/gold/aspect_sentences/`) contains punctuation-preserving review sentences with product, time, rating, and helpfulness context for later lexicon/LDA/aspect-sentiment work.

The medallion jobs do not train models. Future training must fit text features on the training split only and may balance only that training split; validation and test distributions remain realistic. Aspect detection and aspect sentiment scoring are also downstream analytical/modeling stages rather than ingestion transformations.
