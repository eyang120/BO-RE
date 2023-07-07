from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np

st.title('Introduction to building Streamlit WebApp')
st.sidebar.title('This is the sidebar')

"""
Correct the data types of the BO RE data. Transform Timestamp into datetime format and force all numerical data.
"""
def correct_dtypes(df: pd.DataFrame):
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

"""
General dataframe cleaning: 
- Drop rows were all values are NaN/NaT
- Drop columns with excessive NaN
- Interpolate missing values
- Remove outliers
- Set timestamp as index
"""
def gen_cleaning(df: pd.DataFrame):
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
    st.subheader("Show Bogalusa Reduction Efficiency dataset")
    st.write(df)