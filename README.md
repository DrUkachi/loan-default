# Loan Default Prediction — End-to-End ML Pipeline

An end-to-end machine learning project that predicts whether a loan applicant will default. The pipeline covers data ingestion, cleaning, validation, model training, evaluation, and deployment, all orchestrated with [MLflow](https://mlflow.org/) and tracked with [Weights & Biases (W&B)](https://wandb.ai/).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Running Individual Steps](#running-individual-steps)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Adding a New Pipeline Step](#adding-a-new-pipeline-step)
- [License](#license)

---

## Overview

This project builds a production-ready loan default classifier using the German Credit dataset. The goals are:

- Automate the full ML lifecycle (data → model → deployment) in a reproducible way.
- Track every experiment, artifact, and metric in Weights & Biases.
- Serve predictions through a FastAPI REST API and a Streamlit web interface.
- Enable monitoring via Grafana dashboards backed by a PostgreSQL database.

**Target variable**: `default` (binary — `0` = no default, `1` = default)

**Primary model**: Random Forest Classifier

**Key metrics**: Accuracy, weighted F1 score, AUC-ROC

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       MLflow + Hydra (Orchestration)            │
│                                                                 │
│  ┌──────────┐   ┌───────────────┐   ┌───────────┐              │
│  │ get_data │──▶│basic_cleaning │──▶│data_check │              │
│  └──────────┘   └───────────────┘   └───────────┘              │
│                                           │                     │
│                                           ▼                     │
│                               ┌──────────────────┐             │
│                               │  train_val_test   │             │
│                               │      _split       │             │
│                               └──────────────────┘             │
│                                           │                     │
│                                           ▼                     │
│                               ┌──────────────────┐             │
│                               │ train_random_    │             │
│                               │    forest        │             │
│                               └──────────────────┘             │
│                                           │                     │
│                          (promote to prod)│                     │
│                                           ▼                     │
│                               ┌──────────────────┐             │
│                               │test_classification│             │
│                               │     _model       │             │
│                               └──────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                   │ Artifacts & metrics tracked in W&B │

                        ┌─────────────────────┐
                        │   Deployment        │
                        │  FastAPI  +         │
                        │  Streamlit UI       │
                        │  (Docker / Compose) │
                        └─────────────────────┘
```

---

## Dataset

The project uses the **German Credit** dataset (`ml_components/get_data/data/credit.csv`), derived from the [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) dataset.

| Feature | Type | Description |
|---|---|---|
| `checking_balance` | Numeric | Current checking account balance |
| `months_loan_duration` | Numeric | Loan duration in months |
| `credit_history` | Categorical | Past credit behaviour (`critical`, `repaid`, `delayed`, etc.) |
| `purpose` | Categorical | Reason for the loan (`radio/tv`, `education`, `car (new)`, etc.) |
| `amount` | Numeric | Loan amount |
| `savings_balance` | Numeric | Savings account balance |
| `employment_length` | Categorical | Length of current employment |
| `installment_rate` | Numeric | Installment rate as a percentage of disposable income |
| `personal_status` | Categorical | Marital / gender status (`single`, `married`, `divorced`) |
| `other_debtors` | Categorical | Other debtors or guarantors |
| `residence_history` | Categorical | Years at current address |
| `property` | Categorical | Most valuable property owned |
| `age` | Numeric | Applicant age in years |
| `installment_plan` | Categorical | Other installment plans (`none`, `bank`, `stores`) |
| `housing` | Categorical | Type of housing (`own`, `rent`, `for free`) |
| `existing_credits` | Numeric | Number of existing credits at this bank |
| `default` | Binary | **Target** — `1` = default, `0` = no default |
| `dependents` | Numeric | Number of dependents |
| `telephone` | Categorical | Whether applicant has a telephone (transformed to `has_telephone`) |
| `foreign_worker` | Categorical | Whether applicant is a foreign worker |
| `job` | Categorical | Job category |
| `gender` | Categorical | Applicant gender |

---

## Project Structure

```
.
├── conda.yml                         # Root conda environment (orchestration)
├── config.yaml                       # Hydra configuration for the pipeline
├── environment.yaml                  # Development environment
├── main.py                           # Pipeline entry point
├── MLProject                         # Root MLflow project definition
├── LICENSE
├── README.md
│
├── cookie-mlflow-step/               # Cookiecutter template for new pipeline steps
│   ├── cookiecutter.json
│   └── {{cookiecutter.step_name}}/
│       ├── MLproject
│       └── conda.yml
│
├── ml_components/                    # Reusable, versioned pipeline components
│   ├── conda.yml
│   ├── setup.py
│   ├── get_data/                     # Step 1: ingest raw data → W&B artifact
│   │   ├── MLProject
│   │   ├── conda.yml
│   │   ├── run.py
│   │   └── data/
│   │       └── credit.csv
│   ├── train_val_test_split/         # Step 4: split data into train/val/test sets
│   │   ├── MLProject
│   │   ├── conda.yml
│   │   └── run.py
│   ├── test_classification_model/    # Step 6 (optional): evaluate final model on test set
│   │   ├── MLProject
│   │   ├── conda.yml
│   │   └── run.py
│   └── wandb_utils/                  # Shared W&B helper utilities
│       ├── __init__.py
│       ├── log_artifact.py
│       └── sanitize_path.py
│
├── modelling/                        # Project-specific modelling steps
│   ├── basic_cleaning/               # Step 2: data cleaning
│   │   ├── MLProject
│   │   ├── conda.yml
│   │   └── run.py
│   ├── data_check/                   # Step 3: automated data validation (pytest)
│   │   ├── MLProject
│   │   ├── conda.yml
│   │   ├── conftest.py
│   │   └── test_data.py
│   ├── eda/                          # Exploratory data analysis
│   │   ├── MLProject
│   │   └── conda.yml
│   ├── train_random_forest/          # Step 5a: train Random Forest
│   │   ├── MLProject
│   │   ├── conda.yml
│   │   ├── feature_engineering.py
│   │   └── run.py
│   ├── train_logistic_regression/    # Step 5b: train Logistic Regression
│   │   ├── MLProject
│   │   ├── conda.yml
│   │   ├── feature_engineering.py
│   │   └── run.py
│   └── train_xgboost/                # Step 5c: train XGBoost
│       ├── MLProject
│       ├── conda.yml
│       ├── feature_engineering.py
│       └── run.py
│
└── deployment/                       # Serving layer
    ├── app.py                        # FastAPI REST API
    ├── loan_prediction_ui.py         # Streamlit web UI
    ├── requirements.txt              # Python dependencies
    ├── Dockerfile                    # Docker image definition
    ├── docker-compose.yml            # Full-stack compose (API + Grafana + Postgres)
    ├── config.yaml
    ├── config/                       # Grafana provisioning configs
    └── monitoring/                   # Monitoring notebooks and artifacts
```

---

## Pipeline Steps

The pipeline is composed of six sequential steps. Each step is an independent MLflow project that reads and writes versioned artifacts in W&B.

### Step 1 — `download` (get_data)

Reads the raw credit CSV file from the local `data/` directory and uploads it to W&B as a raw data artifact (`sample.csv`).

### Step 2 — `basic_cleaning`

Downloads `sample.csv` from W&B and applies the following transformations:

- **Categorical imputation**: fills missing categorical values with the mode of each feature calculated within each class of the `default` target column (i.e., the mode for default=0 and default=1 groups separately).
- **Telephone feature**: converts the `telephone` column into a binary `has_telephone` flag and drops the original column.
- **Numerical imputation**: fills missing numerical values with the column mean.
- Outputs `clean_sample.csv` back to W&B.

### Step 3 — `data_check`

Runs automated pytest-based data quality checks against `clean_sample.csv`:

| Test | What it checks |
|---|---|
| `test_column_names` | All 22 expected columns are present and in order |
| `test_installment_plan` | Only valid installment plan categories exist |
| `test_purposes` | Only valid loan purpose categories exist |
| `test_personal_status` | Only valid personal status categories exist |
| `test_similar_loan_duration_distr` | KL divergence of `months_loan_duration` vs reference is below threshold |
| `test_similar_residence_distr` | KL divergence of `residence_history` vs reference is below threshold |

### Step 4 — `data_split` (train_val_test_split)

Splits the cleaned dataset into:
- **`trainval_data.csv`** (80%) — used for training and validation.
- **`test_data.csv`** (20%) — held out for final evaluation.

Both splits are stratified on the `default` column by default.

### Step 5 — `train_random_forest`

Trains a **scikit-learn Random Forest classifier** inside a full inference pipeline:

1. **Numerical preprocessing**: mean imputation for `amount`, `savings_balance`, `age`.
2. **Categorical preprocessing**: most-frequent imputation + one-hot encoding for 11 categorical features.
3. **Classifier**: `RandomForestClassifier` with hyperparameters taken from `config.yaml`.

Metrics logged to W&B:

| Metric | Description |
|---|---|
| `accuracy` | Validation accuracy |
| `f1` | Weighted F1 score |
| `auc_roc` | Area under the ROC curve |

The trained pipeline is exported as an MLflow model artifact (`random_forest_export`) and logged to W&B.

### Step 6 — `test_classification_model` *(manual, run after promotion)*

Downloads the model tagged as **`prod`** in W&B and evaluates it against the held-out test set. Logs accuracy, F1, and AUC-ROC, and uploads a `predictions.csv` artifact.

> **Note**: This step is not included in the default pipeline run. You must first promote a model run to the `prod` alias in W&B, then trigger this step explicitly (see [Running Individual Steps](#running-individual-steps)).

---

## Configuration

All pipeline parameters are controlled via **`config.yaml`** and managed with [Hydra](https://hydra.cc/). Key sections:

```yaml
main:
  project_name: loan_default          # W&B project name
  experiment_name: development        # W&B run group
  steps: all                          # Comma-separated steps, or "all"

etl:
  sample: "credit.csv"                # Source CSV file name

data_check:
  kl_threshold: 0.2                   # Max KL divergence (in bits) before data check fails.
                                      # Lower values enforce stricter distribution matching.
                                      # 0.2 is a reasonable default for this dataset size.

modeling:
  test_size: 0.2                      # Fraction of data held out for testing
  val_size: 0.2                       # Fraction of trainval used for validation
  random_seed: 42
  stratify_by: "default"              # Column to stratify splits on

  random_forest:
    n_estimators: 100
    max_depth: 15
    min_samples_split: 4
    min_samples_leaf: 3
    n_jobs: -1                        # -1 = use all CPU cores
    criterion: log_loss
    max_features: 0.5
    oob_score: true
```

Override any parameter at runtime using Hydra syntax (see examples below).

---

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)
- [MLflow](https://mlflow.org/) — installed via conda environment
- A [Weights & Biases](https://wandb.ai/) account and API key
- (Optional for deployment) [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/DrUkachi/loan-default.git
cd loan-default
```

### 2. Set up the development environment

```bash
conda env create -f environment.yaml
conda activate loan_default_dev
```

### 3. Log in to Weights & Biases

```bash
wandb login
```

### 4. Run the full pipeline

```bash
mlflow run . -P steps=all
```

This executes all steps in sequence: `download → basic_cleaning → data_check → data_split → train_random_forest`.

---

## Running Individual Steps

Run a subset of steps by passing a comma-separated list:

```bash
# Run only the download and cleaning steps
mlflow run . -P steps=download,basic_cleaning

# Run only model training
mlflow run . -P steps=train_random_forest
```

Override configuration parameters at runtime:

```bash
# Change the number of trees and max depth
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.n_estimators=200 modeling.random_forest.max_depth=10"

# Change the test split size
mlflow run . \
  -P hydra_options="modeling.test_size=0.15"
```

### Running the final test evaluation

After promoting a trained model to `prod` in the W&B UI:

```bash
mlflow run . -P steps=test_classification_model
```

---

## Deployment

The `deployment/` directory contains everything needed to serve the model.

### Services

| Service | Description | Port |
|---|---|---|
| **FastAPI** | REST API for single and batch inference | `80` |
| **Streamlit** | Interactive web UI for manual or CSV-based prediction | `8501` |
| **PostgreSQL** | Database for storing predictions and monitoring data | `5432` |
| **Adminer** | Lightweight database management UI | `8080` |
| **Grafana** | Dashboards for model and data monitoring | `3000` |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict default probability for a single applicant (JSON body) |
| `POST` | `/batch-predict` | Predict defaults for a batch of applicants (CSV file upload) |

**Single prediction example:**

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{
    "checking_balance": -43,
    "months_loan_duration": 6,
    "credit_history": "critical",
    "purpose": "radio/tv",
    "amount": 1169,
    "savings_balance": 0,
    "employment_length": "13 years",
    "installment_rate": 4,
    "personal_status": "single",
    "other_debtors": "none",
    "residence_history": "6 years",
    "property": "real estate",
    "age": 67,
    "installment_plan": "none",
    "housing": "own",
    "existing_credits": 2,
    "dependents": 1,
    "has_telephone": 1,
    "foreign_worker": "yes",
    "job": "skilled employee",
    "gender": "male"
  }'
```

### Running with Docker

**Build and run the API + Streamlit UI:**

```bash
cd deployment

# Build the image
docker build -t loan-default-app .

# Run the container (set your W&B API key)
docker run -e WANDB_API_KEY=<your_key> -p 80:80 -p 8501:8501 loan-default-app
```

**Run the full monitoring stack (API + Grafana + Postgres):**

```bash
cd deployment
docker-compose up
```

Services will be available at:
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:80/docs
- Grafana: http://localhost:3000
- Adminer: http://localhost:8080

---

## Monitoring

The deployment stack includes a Grafana-based monitoring setup:

- **PostgreSQL** stores prediction logs and data drift metrics.
- **Grafana** reads from PostgreSQL and renders dashboards for tracking model performance and data distribution shifts over time.
- Monitoring notebooks are available in `deployment/monitoring/`.

---

## Adding a New Pipeline Step

A [Cookiecutter](https://cookiecutter.readthedocs.io/) template is provided to scaffold new steps quickly:

```bash
pip install cookiecutter
cookiecutter cookie-mlflow-step -o modelling
```

You will be prompted for the new step name. The template generates an `MLproject` file and a `conda.yml` pre-configured for MLflow + W&B.

After creating the step:
1. Implement the logic in the generated `run.py`.
2. Add the step name to the `_steps` list in `main.py`.
3. Add any required parameters under `config.yaml`.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
