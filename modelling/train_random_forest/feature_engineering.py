import pandas as pd
import numpy as np

def create_loan_application_features(df):

    # 1. Income-Related Features
    df['co_app_total_income'] = df['applicantincome'] + df['coapplicantincome']
    df['income_to_loan_ratio'] = df['total_income'] / df['loanamount']
    df['debt_to_income_ratio'] = (df['applicantincome'] + df['coapplicantincome']) / df['total_income']
    df['income_to_loan_ratio_squared'] = df['income_to_loan_ratio'] ** 2
    df['debt_to_income_ratio_squared'] = df['debt_to_income_ratio'] ** 2
    df['income_to_loan_ratio_cubed'] = df['income_to_loan_ratio'] ** 3
    df['debt_to_income_ratio_cubed'] = df['debt_to_income_ratio'] ** 3

    # 2. Loan Term Feature Engineering
    df['emi'] = df['loanamount'] / df['loan_amount_term']
    df['loan_term_years'] = df['loan_amount_term'] / 12  # Assuming 12 months in a year
    df['loan_term_squared'] = df['loan_amount_term'] ** 2
    df['loan_term_cubed'] = df['loan_amount_term'] ** 3


    # 5. Dependent Information
    df['income_per_dependent'] = df['total_income'] / (df['dependents'] + 1)  # Adding 1 to avoid division by zero
    df['loan_per_dependent'] = df['loanamount'] / (df['dependents'] + 1)
    df['loan_term_per_dependent'] = df['loan_amount_term'] / (df['dependents'] + 1)
    df['income_per_dependent_squared'] = df['income_per_dependent'] ** 2
    df['loan_per_dependent_squared'] = df['loan_per_dependent'] ** 2
    df['loan_term_per_dependent_squared'] = df['loan_term_per_dependent'] ** 2

    # 6. Interaction and Polynomial Features
    df['income_x_property_area'] = df['total_income'] * df['property_area']  # Assuming 'Urban' is a one-hot column
    df['income_squared'] = df['total_income'] ** 2
    df['loan_squared'] = df['loanamount'] ** 2

    return df


def ordinal_experience_ranking(df, column_name):
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

# Example usage
data = {
    'experience': ['13 years', '2 years', '5 years', '3 years', '11 years', '4 years',
                   np.nan, '6 months', '5 months', '16 years', '1 years', '17 years',
                   '3 months', '9 years', '4 months', '10 years', '10 months',
                   '1 months', '7 months', '19 years', '7 years', '14 years',
                   '18 years', '0 months', '15 years', '9 months', '6 years',
                   '8 years', '12 years', '11 months', '2 months', '8 months']
}
df = pd.DataFrame(data)

# Apply the function
df['experience_ordinal'] = ordinal_experience_ranking(df, 'experience')

print(df[['experience', 'experience_ordinal']])
