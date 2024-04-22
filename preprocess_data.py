import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df):
    # Drop the Timestamp column if it still exists
    if 'Timestamp' in df.columns:
        df = df.drop('Timestamp', axis=1)
    
    # Explicitly convert all numeric columns to float to avoid int64 issues
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.astype(float))

    # Fill missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Define and encode categorical columns
    categorical_cols = [
        'Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type',
        'Vehicle_Dimensions', 'Geographical_Location', 'Vehicle_Plate_Number'
    ]
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('category'))
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out())
    
    # Combine non-categorical and encoded data
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    return df
