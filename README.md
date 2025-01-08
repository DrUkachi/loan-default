# End-to-End ML Workflow for an Automated Loan Default Prediction

This repository contains an end-to-end machine learning project aimed at predicting loan defaults. The project includes data collection, data cleaning, exploratory data analysis (EDA), model training, and evaluation.

It uses Weights and Bias and ML Workflow to create the End-to-End workflow

## Project Structure
.
├── conda.yml
├── config.yaml
├── cookie-mlflow-step/
│   ├── {{cookiecutter.step_name}}/
│   │   ├── conda.yml
│   │   ├── MLproject
│   ├── cookiecutter.json
├── environment.yaml
├── LICENSE
├── main.py
├── ml_components/
│   ├── conda.yml
│   ├── get_data/
│   │   ├── conda.yml
│   │   ├── data/
│   │   ├── MLProject
│   │   ├── run.py
│   ├── setup.py
│   ├── test_classification_model/
│   │   ├── conda.yml
│   │   ├── MLProject
│   │   ├── run.py
│   ├── train_val_test_split/
│   │   ├── conda.yml
│   │   ├── MLProject
│   │   ├── run.py
│   ├── wandb_utils/
├── MLProject
├── modelling/
│   ├── basic_cleaning/
│   │   ├── MLProject
│   │   ├── run.py
│   ├── data_check/
│   │   ├── MLProject
│   ├── eda/
│   │   ├── MLProject
│   ├── train_logistic_regression/
│   ├── train_random_forest/
│   │   ├── MLProject
│   ├── train_xgboost/
├── deployment/                   # NEW: Deployment directory
│   ├── app.py                    # FastAPI app code
│   ├── requirements.txt          # Deployment-specific dependencies
│   ├── Dockerfile                # (Optional) Dockerfile for containerization
│   └── config.yaml               # Deployment configuration (e.g., model path)
├── README.md
