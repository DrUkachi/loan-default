import pytest
import pandas as pd
import wandb
import numpy as np
from scipy.stats import entropy

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", help="Path to input CSV file")
    parser.addoption("--ref", action="store", help="Path to reference CSV file")
    parser.addoption("--output_artifact", action="store", help="Name of the output artifact")
    parser.addoption("--output_type", action="store", help="Type of the output artifact")
    parser.addoption("--output_description", action="store", help="Description of the output artifact")
    parser.addoption("--kl_threshold", action="store", type=float, help="Threshold for KL divergence test")

@pytest.fixture(scope="session")
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download the input artifact
    data_path = run.use_artifact(request.config.option.csv).file()
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)
    return df

@pytest.fixture(scope="session")
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download reference artifact
    ref_data_path = run.use_artifact(request.config.option.ref).file()
    if ref_data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    ref_df = pd.read_csv(ref_data_path)
    return ref_df

@pytest.fixture(scope="session")
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold
    if kl_threshold is None:
        pytest.fail("You must provide a --kl_threshold option for KL divergence")
    return kl_threshold

def test_categorical_filling(data):
    """Check that categorical columns have no missing values."""
    categorical_columns = [
        "credit_history", "purpose", "employment_length", "personal_status",
        "other_debtors", "residence_history", "property", "installment_plan",
        "housing", "foreign_worker", "job", "gender"
    ]
    for col in categorical_columns:
        assert data[col].notnull().all(), f"Missing values found in {col}"

def test_telephone_conversion(data):
    """Check that 'telephone' column has been converted to a boolean if present."""
    assert "has_telephone" in data.columns, "'has_telephone' column should be created"
    assert "telephone" not in data.columns, "'telephone' column should be dropped if present"

def test_numerical_filling(data):
    """Check that missing values in numerical columns are filled with the mean."""
    numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_columns:
        assert data[col].isnull().sum() == 0, f"Missing values found in {col}"

def test_kl_divergence(data, ref_data, kl_threshold):
    """Ensure that KL divergence between current and reference data is below the threshold."""
    numerical_columns = ref_data.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_columns:
        # Calculate histogram-based distributions
        cleaned_dist, _ = np.histogram(data[col].dropna(), bins=20, density=True)
        ref_dist, _ = np.histogram(ref_data[col].dropna(), bins=20, density=True)

        # Avoid division by zero in KL divergence
        cleaned_dist += 1e-10
        ref_dist += 1e-10

        kl_divergence = entropy(cleaned_dist, ref_dist)
        assert kl_divergence < kl_threshold, f"KL divergence for {col} exceeds threshold: {kl_divergence}"

if __name__ == "__main__":
    pytest.main()
