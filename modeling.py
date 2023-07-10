import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import matplotlib.pyplot as plt
import plotly.express as px
import mpld3
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

def correct_dtypes(df: pd.DataFrame):
    """
    Correct the data types of the BO RE data. Transform Timestamp into datetime format and force all numerical data.
    """
    if type(df) != pd.DataFrame:
        raise TypeError("Data is not DataFrame!")
    
    numeric_types = (int, float)
    
    for col in df:
        if col == "Timestamp":
            df[col] = pd.to_datetime(df[col])
        elif df[col].dtype not in numeric_types:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def correct_dtypes(df: pd.DataFrame):
    """
    Correct the data types of the BO RE data. Transform Timestamp into datetime format and force all numerical data.
    """
    if type(df) != pd.DataFrame:
        raise TypeError("Data is not DataFrame!")
    
    numeric_types = (int, float)
    
    for col in df:
        if col == "Timestamp":
            df[col] = pd.to_datetime(df[col])
        elif df[col].dtype not in numeric_types:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def gen_cleaning(df: pd.DataFrame):
    """
    General dataframe cleaning: 
    - Drop rows were all values are NaN/NaT
    - Drop columns with excessive NaN
    - Interpolate missing values
    - Remove outliers
    - Set timestamp as index
    """
    df.dropna(inplace=True, how = 'all')
    blank_cols = df.isnull().sum()[df.isnull().sum() > 10].index
    #display(blank_cols)
    df.drop(blank_cols, axis=1, inplace=True)
    df = df.interpolate()
    df = df[df['RE test'] >= 60]
    df.set_index('Timestamp', inplace=True)
    return df
