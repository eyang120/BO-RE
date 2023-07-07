from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
import plotly.express as px

st.title('Bogalusa Reduction Efficiency Model')
st.sidebar.title('Toggle display')

def correct_dtypes(df: pd.DataFrame):
    """
    Correct the data types of the BO RE data. Transform Timestamp into datetime format and force all numerical data.
    """
    if type(df) != pd.DataFrame:
        raise TypeError("Data is not DataFrame!")
    
    numeric_types = (int, float)
    
    for col in df:
        #print(type(col))
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


@st.cache_data(persist= True)
def load():
    data = pd.read_csv("BORE_v3.csv")
    data = correct_dtypes(data)
    data = gen_cleaning(data)
    return data
df = load()

if st.sidebar.checkbox("Display data", False):
    st.subheader("Bogalusa Reduction Efficiency Dataset")
    st.write(df)

@st.cache_data(persist=True)
def feature_selection(data):
    x = data.drop("RE test", axis = 1)
    y = data["RE test"]

    lasso = Lasso(alpha = 0.3)
    lasso.fit(x, y)
    selected_features = x.columns[abs(lasso.coef_) >= 0.1]
    selected_df = data[selected_features.to_list()]
    return selected_df

selected_df = feature_selection(df)

if st.sidebar.checkbox("Display LASSO-selected features", False):
    st.subheader("Selected BO-RE Features:")
    st.write(selected_df)
