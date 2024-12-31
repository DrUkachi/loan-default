#!/usr/bin/env python
"""
[An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and saves the results in Weights & Biases
"""
import os
import argparse
import logging

import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    # Initialize a W&B run
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read the input artifact
    df = pd.read_csv(artifact_local_path)

    # Step 1: Fill missing categorical values based on mode per target group
    categorical_columns = [
        "credit_history", "purpose", "employment_length", "personal_status",
        "other_debtors", "residence_history", "property", "installment_plan",
        "housing", "foreign_worker", "job", "gender"
    ]
    target_col = "default"

    for col in categorical_columns:
        # Calculate the mode for each target group
        modes = df.groupby(target_col)[col].agg(
            lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        )

        # Apply the mode to fill missing values in each group
        for target_value in modes.index:
            df.loc[(df[target_col] == target_value) & (df[col].isnull()), col] = modes[target_value]

    # Step 2: Create a boolean column for 'has_telephone' and drop the 'telephone' column
    df['has_telephone'] = df["telephone"].apply(lambda x: 1 if pd.notnull(x) else 0)
    df.drop(columns=["telephone"], inplace=True)

    # Step 3: Fill missing values in numerical columns with the mean of each column
    numerical_columns = df.select_dtypes(include=['number']).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    # Save cleaned data to a file
    filename = "clean_sample.csv"
    expected_columns = [
        "checking_balance",
        "months_loan_duration",
        "credit_history",
        "purpose",
        "amount",
        "savings_balance",
        "employment_length",
        "installment_rate",
        "personal_status",
        "other_debtors",
        "residence_history",
        "property",
        "age",
        "installment_plan",
        "housing",
        "existing_credits",
        "default",
        "dependents",
        "has_telephone",
        "foreign_worker",
        "job",
        "gender"
    ]
    df = df[expected_columns]
    df.to_csv(filename, index=False)

    # Log the artifact to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # Clean up local file
    os.remove(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="This is the file name of the input artifact",
        required=True
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="This is the file name of the output artifact",
        required=True
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="This is the type for the output artifact created",
        required=True
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="This is the description of the output artifact that will be created and stored",
        required=True
    )

    args = parser.parse_args()
    go(args)
