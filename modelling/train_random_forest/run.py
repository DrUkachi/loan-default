#!/usr/bin/env python
"""
This script trains a Random Forest on the provided credit dataset.
"""
import argparse
import logging
import os
import shutil
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

import mlflow
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Load Random Forest configuration
    try:
        with open(args.rf_config) as fp:
            rf_config = json.load(fp)
    except Exception as e:
        logger.error(f"Failed to load random forest config: {e}")
        return

    run.config.update(rf_config)
    rf_config["random_state"] = args.random_seed

    # Load dataset
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    data = pd.read_csv(trainval_local_path)
    logger.info(f"Dataset loaded with shape {data.shape}")

    # Define target and features
    target = "default"
    y = data.pop(target)
    X = data

    # Train-test split
    stratify = y if args.stratify_by != "none" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=stratify, random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")
    sk_pipe, processed_features = get_inference_pipeline(rf_config)

    logger.info("Fitting the pipeline")
    sk_pipe.fit(X_train, y_train)

    logger.info("Evaluating the model")
    y_pred = sk_pipe.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    logger.info(f"Validation Accuracy: {accuracy}")

    weighted_f1 = f1_score(y_val, y_pred, average="weighted")
    logger.info(f"Validation F1: {weighted_f1}")

    y_prob = sk_pipe.predict_proba(X_val)[:, 1]
    auc_roc = roc_auc_score(y_val, y_prob)
    logger.info(f"Validation AUC-ROC: {auc_roc}")

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    # Log metrics and artifact
    run.summary.update({"accuracy": accuracy, "f1": weighted_f1, "auc_roc": auc_roc})
    run.log({"accuracy": accuracy, "f1": weighted_f1, "auc_roc": auc_roc})
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})

    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_val, y_pred))

    # Export the model
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(sk_pipe, "random_forest_dir")

    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random Forest model export",
        metadata=rf_config,
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)]
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp

def get_inference_pipeline(rf_config):
    numerical_features = ["amount", "savings_balance", "age"]
    numerical_transformer = SimpleImputer(strategy="mean")

    categorical_features = [
        "credit_history", "purpose", "employment_length", "personal_status",
        "other_debtors", "residence_history", "property", "installment_plan",
        "housing", "job", "gender"
    ]
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    random_forest = RandomForestClassifier(**rf_config)

    sk_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("random_forest", random_forest),
    ])

    processed_features = numerical_features + categorical_features
    return sk_pipe, processed_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Random Forest on credit dataset"
        )
    
    parser.add_argument(
        "--trainval_artifact", 
        type=str, 
        required=True,
        help="Artifact containing the training dataset. It will be split into train and validation"
        )

    parser.add_argument(
        "--val_size", 
        type=float, 
        default=0.2,
        help="Fraction of the dataset to use for validation",
        )

    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--stratify_by", 
        type=str, 
        default="none", 
        required=False,
        help="Column to use for stratification"
        )

    parser.add_argument(
        "--rf_config", 
        type=str,
        required=True,
        default="{}",
        help="""
        Random forest configuration. A JSON dict that will be passed to thescikit-learn constructor for RandomForestRegressor.
        """
        )
    
    parser.add_argument(
        "--output_artifact", 
        type=str,
        required=True,
        help="Name for the output serialized model artifact")

    args = parser.parse_args()
    go(args)
