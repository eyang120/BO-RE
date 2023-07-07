
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import matplotlib.pyplot as plt
import plotly.express as px
import mpld3
import streamlit.components.v1 as components

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
x = df.drop("RE test", axis = 1)
y = df["RE test"]

if st.sidebar.checkbox("Display data", False):
    st.subheader("Bogalusa Reduction Efficiency Dataset")
    st.write(df)

@st.cache_data(persist=True)
def feature_selection(data, x, y):
    lasso = Lasso(alpha = 0.3)
    lasso.fit(x, y)
    selected_features = x.columns[abs(lasso.coef_) >= 0.1]
    selected_df = data[selected_features.to_list()]
    return selected_df

selected_df = feature_selection(df, x, y)

if st.sidebar.checkbox("Display LASSO-selected features", False):
    st.subheader("Selected BO-RE Features:")
    st.write(selected_df)

@st.cache_data(persist=True)
def model_building(x2, y):
    x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.3)

    lr = LinearRegression()

    cv_rmse_scores = -cross_val_score(lr, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = cross_val_score(lr, x_train, y_train, scoring='r2', cv=5)

    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)

    fig = plt.figure()
    plt.scatter(y_pred, y_test, color='blue', label="Actual")
    plt.title("Linear Regression - Actual vs Predicted")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.legend()
    st.pyplot(fig)

    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_rmse_scores))
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_rmse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores)
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)

if st.button("Run Model"):
    model_building(selected_df, y)