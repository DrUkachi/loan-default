#!/usr/bin/env python
"""
[An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
"""
import os
import argparse
import logging

import wandb
import pandas as pd



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")

    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Step 1: Fill in the categorical variables
  # Fill missing categorical values based on mode per target group
    categorical_columns = [
    "credit_history", "purpose", "employment_length", "personal_status",
    "other_debtors", "residence_history", "property", "installment_plan",
    "housing", "foreign_worker", "job", "gender"
    ]

    target_col = "default"

    for col in categorical_columns:
        # Calculate the mode for each target group
        modes = df.groupby(target_col)[col].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')

        # Apply the mode to fill missing values in each group
        for target_value in modes.index:
            df.loc[(df[target_col] == target_value) & (df[col].isnull()), col] = modes[target_value]

    # Step 2:
    # Since a majority of these values are we missing we can choose to drop it entirely or convert to a boolean column.
    # If its not important we can later drop it through the feature selection process.
    df['has_telephone'] = df["telephone"].apply(lambda x: 1 if pd.notnull(x) and x != '' else 0)
    df.drop(columns=["telephone"], inplace=True)

    # Step 3
    # Fill missing values in all numerical columns with the mean of each column
    df.fillna(df.mean(), inplace=True)

    filename = "clean_sample.csv"

    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help= "This is the file name of the input artifact",
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
