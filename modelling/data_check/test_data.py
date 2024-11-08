import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_colums = [
        "months_loan_duration",
        "credit_history",
        "purpose",
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
        "telephone",
        "foreign_worker",
        "job",
        "gender"
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_installment_plan(data):

    known_installments = ['none', 'bank', 'stores']

    installments = set(data['installment_plan'].unique())

    # Unordered check
    assert set(known_installments) == set(installments)

def test_purposes(data):
    known_purposes = ['radio/tv', 'education', 'furniture', 'car (new)', 'car (used)',
       'business', 'domestic appliances', 'repairs', 'others',
       'retraining']
    
    purposes = set(data['purpose'].unique())

    # Unordered check
    assert set(known_purposes) == set(purposes)

def test_personal_status(data):
    known_status = ['single', 'divorced', 'married']

    status = set(data['personal_status'].unique())

    # Unorded check
    assert set(known_status) ==  set(status)


def test_similar_age_distr(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['age'].value_counts().sort_index()
    dist2 = ref_data['age'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold