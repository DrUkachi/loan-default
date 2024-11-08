#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline

def ordinal_ranking(df, column_name):
    """
    Convert experience data in years and months to ordinal rankings in a DataFrame column.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: str, name of the column to convert to ordinal ranking.
    
    Returns:
    - pandas Series with ordinal ranks.
    """
    # Helper function to convert each entry to months
    def convert_to_months(value):
        if pd.isna(value):
            return np.nan
        elif 'years' in value:
            return int(value.split()[0]) * 12  # Convert years to months
        elif 'months' in value:
            return int(value.split()[0])       # Already in months
    
    # Apply the conversion to months
    df[f'{column_name}_in_months'] = df[column_name].apply(convert_to_months)
    
    # Generate ordinal ranking based on months, ignoring NaN values
    df[f'{column_name}_ordinal_rank'] = df[f'{column_name}_in_months'].rank(method="min")
    
    return df[f'{column_name}_ordinal_rank']


def create_loan_application_features(df):

    # 1. Balance to Loan Ratio
    df['total_balance'] = df['checking_balance'] + df['savings_balance']
    df['total_balance_to_loan_ratio'] = df['total_balance'] / df['amount']
    df['checking_balance_to_loan_ratio'] = df['checking_balance'] / df['amount']
    df['savings_balance_to_loan_ratio'] = df['savings_balance'] / df['amount']
    df['balance_to_loan_ratio_squared'] = df['total_balance_to_loan_ratio'] ** 2
    df['balance_to_loan_ratio_cubed'] = df['balance_to_loan_ratio_cubed'] ** 3

    # 2. Loan Term Feature Engineering
    df['emi'] = df['amount'] / df['months_loan_duration'] # equated monthly installments
    df['loan_term_years'] = df['months_loan_duration'] / 12  # Assuming 12 months in a year
    df['loan_term_squared'] = df['months_loan_duration'] ** 2
    df['loan_term_cubed'] = df['months_loan_duration'] ** 3


    # 3. Dependent Information
    df['income_per_dependent'] = df['total_balance'] / (df['dependents'] + 1)  # Adding 1 to avoid division by zero
    df['loan_per_dependent'] = df['amount'] / (df['dependents'] + 1)
    df['loan_term_per_dependent'] = df['months_loan_duration'] / (df['dependents'] + 1)
    df['income_per_dependent_squared'] = df['income_per_dependent'] ** 2
    df['loan_per_dependent_squared'] = df['loan_per_dependent'] ** 2
    df['loan_term_per_dependent_squared'] = df['loan_term_per_dependent'] ** 2

    # 4. Credit Information
    df['exisiting_credits_per_loan'] = df['existing_credits'] / df['amount']
    df['installment_rate_per_credits'] = df['installment_rate'] / df['existing_credits']
    df['balance_per_credit_information'] = df['total_balance'] / df['existing_credits']


    # Creating Ordinal Features
    df['employment_length_ordinal'] = ordinal_ranking(df, "employment_length")
    df['residence_history_ordinal'] = ordinal_ranking(df, "residence_history")
    df.drop(columns=["employment_length", "residence_history"], inplace=True)

    # 5. Interaction and Polynomial Features
    df['age_X_dependent'] = df['age'] * df['dependents']
    df['income_squared'] = df['total_balance'] ** 2
    df['loan_squared'] = df['amount'] ** 2
    df['income_x_employment_length'] = df['total_balance'] * df['employment_length_ordinal']
    df['income_x_residence_history'] = df['total_balance'] * df['residence_history_ordinal']

    return df



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    
    # Use run.use_artifact(...).file() to get the train and validation artifact (args.trainval_artifact)
    # and save the returned path in train_local_pat
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    

    X = pd.read_csv(trainval_local_path)
    y = X.pop("default")  # this removes the column "price" from X and puts it into y

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by] if args.stratify_by != 'none' else None, random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")


    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    
    sk_pipe.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Scoring")

    y_pred = sk_pipe.predict(X_val)
    y_prob = sk_pipe.predict_proba(X_val)[:, 1] # Get probabilities for AUC-ROC

    # Calculate accuracy, F1 score and AUC_ROC
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="binary")  # 'weighted' for multiclass, 'binary' for binary
    
    auc_roc = roc_auc_score(y_val, y_prob)

    logger.info(f"Accuracy Score: {accuracy}")
    logger.info(f"MAE: {f1}")
    logger.info(f"AUC ROC Score: {auc_roc}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    
    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"
    # HINT: use mlflow.sklearn.save_model
    # YOUR CODE HERE
    mlflow.sklearn.save_model(sk_pipe, "random_forest_dir")

    
    # Upload the model we just exported to W&B
    # HINT: use wandb.Artifact to create an artifact. Use args.output_artifact as artifact name, "model_export" as
    # type, provide a description and add rf_config as metadata. Then, use the .add_dir method of the artifact instance
    # you just created to add the "random_forest_dir" directory to the artifact, and finally use
    # run.log_artifact to log the artifact to the run
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random Forest model export",
        metadata=rf_config
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    ######################################
    # Here we save r_squared under the "r2" key
    run.summary['accuracy'] = accuracy
    # Now log the variable "mae" under the key "mae".
    run.summary['f1'] = f1

    run.summary['auc-roc'] = auc_roc

    ######################################

    # Upload to W&B the feture importance visualization
    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    # idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config):
    # Let's handle the categorical features first
    # Ordinal categorical are categorical values for which the order is meaningful, for example
    # for room type: 'Entire home/apt' > 'Private room' > 'Shared room'
    # ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["credit_history",
                               "purpose",
                               "personal_status",
                               "other_debtors",
                               "property",
                               "installment_plan",
                               "housing", 
                               "foreign_worker", 
                               "job", 
                               "gender"]
    
    #  ordinal_categorical_preproc = OrdinalEncoder()

   
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )
    

    # Let's impute the numerical columns to make sure we can handle missing values
    # (note that we do not scale because the RF algorithm does not need that)
    zero_imputed = [
        ['checking_balance', 'months_loan_duration', 'amount', 'savings_balance',
       'installment_rate', 'age']
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),

        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    # Create random forest
    random_Forest = RandomForestRegressor(**rf_config)

    ######################################
    # Create the inference pipeline. The pipeline must have 2 steps: a step called "preprocessor" applying the
    # ColumnTransformer instance that we saved in the `preprocessor` variable, and a step called "random_forest"
    # with the random forest instance that we just saved in the `random_forest` variable.
    # HINT: Use the explicit Pipeline constructor so you can assign the names to the steps, do not use make_pipeline
    sk_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("random_forest", random_Forest)
    ])

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )


    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
