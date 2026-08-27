"""Submit the Gold-layer sentiment training job to Amazon SageMaker AI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_ROLE_ARN"),
        help="SageMaker execution-role ARN (or set SAGEMAKER_ROLE_ARN)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("GLUE_S3_BUCKET", "amazon-food-reviews-ml-model"),
    )
    parser.add_argument("--gold-prefix", default="gold")
    parser.add_argument("--model-prefix", default="models")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit without streaming logs until training completes",
    )
    args = parser.parse_args()

    if not args.role_arn:
        parser.error("provide --role-arn or set SAGEMAKER_ROLE_ARN")
    if not args.region:
        parser.error("provide --region or configure an AWS default region")
    return args


def s3_uri(bucket: str, prefix: str) -> str:
    return f"s3://{bucket}/{prefix.strip('/')}/"


def main() -> None:
    args = parse_args()
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    gold_uri = s3_uri(args.bucket, args.gold_prefix)
    model_uri = s3_uri(args.bucket, args.model_prefix)
    source_dir = Path(__file__).resolve().parent

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(source_dir),
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="1.4-2",
        py_version="py3",
        output_path=model_uri,
        base_job_name="amazon-reviews-sentiment",
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "max-features": args.max_features,
            "test-size": args.test_size,
            "c": args.c,
        },
        metric_definitions=[
            {"Name": "validation:accuracy", "Regex": "validation:accuracy=([0-9.]+)"},
            {"Name": "validation:precision", "Regex": "validation:precision=([0-9.]+)"},
            {"Name": "validation:recall", "Regex": "validation:recall=([0-9.]+)"},
            {"Name": "validation:f1", "Regex": "validation:f1=([0-9.]+)"},
        ],
        max_run=3_600,
        volume_size=10,
        tags=[{"Key": "Project", "Value": "amazon-reviews-sentiment"}],
    )

    training_input = TrainingInput(
        s3_data=gold_uri,
        s3_data_type="S3Prefix",
        input_mode="File",
        content_type="application/x-parquet",
    )

    print(f"Gold input: {gold_uri}")
    print(f"Model output: {model_uri}")
    estimator.fit({"train": training_input}, wait=not args.no_wait)

    print(f"Training job: {estimator.latest_training_job.name}")
    if not args.no_wait:
        print(f"Model artifact: {estimator.model_data}")


if __name__ == "__main__":
    main()
