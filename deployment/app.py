from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import wandb
import joblib
from typing import Dict

# Initialize FastAPI
app = FastAPI()

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler for the FastAPI app. Handles startup and shutdown events.
    """
    global model
    try:
        # Initialize a W&B API client and load the model during startup
        wandb_api = wandb.Api()

        # Fetch the artifact
        artifact = wandb_api.artifact("loan_default/random_forest_export:prod", type="model_export")
        model_dir = artifact.download()  # Downloads the artifact to a local directory

        # Load the model (adjust based on your model's format)
        model_path = f"{model_dir}/model.pkl"  # Replace with the actual model file name
        model = joblib.load(model_path)
        print("Model loaded successfully.")

        yield  # Continue the app's lifespan

    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# Set the lifespan event handler for the app
app = FastAPI(lifespan=lifespan)
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan app API!"}

@app.post("/predict")
def predict(data: Dict):
    """
    Perform inference using the loaded model.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data])

        # Perform prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]  # Assuming a classification task

        return {"prediction": prediction.tolist(), "probability": probability.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Perform batch inference using a CSV file.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(file.file)

        # Validate that the necessary columns are present
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
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
        
        # Perform batch prediction
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]  # Assuming a classification task

        # Return predictions and probabilities
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during batch prediction: {e}")
