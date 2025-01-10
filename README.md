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
├── deployment/                   
│   ├── app.py                    
│   ├── requirements.txt          
│   ├── Dockerfile                
│   └── config.yaml               
├── README.md

To run the Dockerfile, follow these steps:

1. **Build the Docker image**:
   Open a terminal and navigate to the directory containing your Dockerfile. Run the following command to build the Docker image:
   ```sh
   docker build -t loan-default-app .
   ```

2. **Run the Docker container**:
   After the image is built, run the container using the following command:
   ```sh
   docker run -p 80:80 -p 8501:8501 loan-default-app
   ```
