import pandas as pd
import numpy as np

def handle_missing_data(df):
    """Impute missing values in the DataFrame using median for numeric columns and mode for categorical columns."""
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Handle numeric columns
            df[column].fillna(df[column].median(), inplace=True)
        else:
            # For categorical columns, replace empty strings with NaN and then fill with the most frequent value (mode)
            df[column].replace({' ': np.nan, '': np.nan}, inplace=True)  # Also consider empty strings as missing values
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def clean_and_prepare_data(df):
    """Convert categorical variables to numeric and perform any additional cleaning."""
    # Map Yes/No columns to 1/0
    yes_no_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

    # Map furnishingstatus to 1 (furnished), 2 (unfurnished), 3 (semi furnished)
    if 'furnishingstatus' in df.columns:
        df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 1, 'unfurnished': 2, 'semi furnished': 3}).fillna(0)

    # Ensure all remaining object-type columns are converted if not already handled above
    for column in df.select_dtypes(include=['object']).columns:
        unique_values = df[column].unique()
        df[column] = pd.Categorical(df[column], categories=unique_values).codes

    return df

def basic_validation(df):
    """Perform basic data validation checks."""
    if df.isnull().any().any():
        raise ValueError("Data still contains null values after attempting to handle them.")
    if df.select_dtypes(include=['object']).any().any():
        raise ValueError("Data still contains non-numeric values which might cause errors in modeling.")
    print("Data validation passed. No missing values and all data is numeric.")

if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('data/Housing.csv')
    df = handle_missing_data(df)
    df = clean_and_prepare_data(df)
    basic_validation(df)
    df.to_csv('data/Cleaned_Housing.csv', index=False)
    print("Cleaned data saved to 'data/Cleaned_Housing.csv'.")
