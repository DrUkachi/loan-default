#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("default")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)
    y_prob = sk_pipe.predict_proba(X_test)[:, 1] # Get probabilities for AUC-ROC

    logger.info("Scoring")
    # Calculate accuracy, F1 score and AUC_ROC
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")  # 'weighted' for multiclass, 'binary' for binary
    
    auc_roc = roc_auc_score(y_test, y_prob)

    logger.info(f"Accuracy Score: {accuracy}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"AUC ROC Score: {auc_roc}")

    # Log MAE and r2
    run.summary['accuracy'] = accuracy
    run.summary['f1'] = f1
    run.summary['auc-roc'] = auc_roc

    # Log model predictions as an artifact
    preds_df = pd.DataFrame({"predictions": y_pred, "actuals": y_test})
    preds_csv_path = "predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)

    artifact = wandb.Artifact(
        "model_predictions",
        type="predictions",
        description="Model predictions and actual values",
    )
    artifact.add_file(preds_csv_path)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str, 
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
