# src/ai_core/data_pipeline.py

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load CSV file and return a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, date_col, target_col, categorical_cols=None):
    """
    Preprocess the data:
    - Ensure date column is datetime
    - Sort by date
    - Create basic temporal features: day_of_week, month
    - Optionally encode categorical columns for ML models
    - Standardize target and date columns to 'y' and 'ds' for forecasting
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        date_col (str): Name of the date column
        target_col (str): Name of the target column
        categorical_cols (list of str, optional): List of categorical columns to encode
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Temporal features
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    
    # Encode categorical features if provided
    if categorical_cols:
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes

    # Standardize columns for forecasting
    df = df.rename(columns={target_col: 'y', date_col: 'ds'})
    
    return df