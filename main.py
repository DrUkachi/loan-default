import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file from a URL and load in W&B - (Try this out)
            # Extract from a database and load in W&B - 
            # Extract file from the local directory and load in W&B - (This is what was implemented)
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            
             _ = mlflow.run(
                    os.path.join(hydra.utils.get_original_cwd(), "modelling", "basic_cleaning"),
                        "main",
                        parameters={
                            "input_artifact": "sample.csv:latest",
                            "output_artifact": "clean_sample.csv",
                            "output_type": "clean_sample",
                            "output_description": "Data is now cleaned and ready for use."
                        },
            )
            

        if "data_check" in active_steps:
            
            _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "modelling", "data_check"),
            "main",
            parameters={
                "csv": "clean_sample.csv:latest",
                "ref": "clean_sample.csv:reference",
                "kl_threshold": config["data_check"]["kl_threshold"],
            },
        )
        
        if "feature_engineering" in active_steps:

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "modelling", "feature_engineering"),
                "main",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "output_artifact": "processed_sample.csv",
                    "output_description": "Data has now being processed"
                }
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            "main",
            parameters={

                "input": "processed_sample.csv",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
        
            },
        )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step


            _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "modelling", "train_random_forest"),
            "main",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config["modeling"]["val_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
                "rf_config": rf_config,
                "output_artifact": "random_forest_export",
                },
            )

        if "test_regression_model" in active_steps:

            _ = mlflow.run(
            f"{config['main']['components_repository']}/test_regression_model",
            "main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_dataset": "test_data.csv:latest"
            },
        )




if __name__ == "__main__":
    go()
