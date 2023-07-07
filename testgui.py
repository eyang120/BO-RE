
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
x_train, x_test, y_train, y_test = train_test_split(selected_df, y, test_size=0.3)


if st.sidebar.checkbox("Display LASSO-selected features", False):
    st.subheader("Selected BO-RE Features:")
    st.write(selected_df)

st.sidebar.subheader("Choose model:")
model_select = st.sidebar.selectbox("Model", ("Linear", "Random Forest", "SVR", "Ridge", "Neutral Net", "XGBoost"))

@st.cache_data(persist=True)
def linear_model(x_train, x_test, y_train, y_test):
    
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
    st.write(np.sqrt(cv_rmse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_rmse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)
    add_sliders(x_train)


def random_forest(x_train, x_test, y_train, y_test):
    rf_regressor = RandomForestRegressor(n_estimators=350)

    cv_mse_scores = -cross_val_score(rf_regressor, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = 1 - (cv_mse_scores / np.var(y_train))

    rf_regressor.fit(x_train, y_train)
    y_pred = rf_regressor.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)

    fig = plt.figure()
    plt.scatter(y_pred, y_test, color='blue', label="Actual")
    plt.title("Random Forest Regression - Actual vs Predicted")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.legend()
    st.pyplot(fig)

    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_mse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_mse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)

def svr_model(x_train, x_test, y_train, y_test):
    svr = SVR(kernel='rbf')

    cv_mse_scores = -cross_val_score(svr, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = cross_val_score(svr, x_train, y_train, scoring='r2', cv=5)

    svr.fit(x_train, y_train)
    y_pred = svr.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)

    fig = plt.figure()
    plt.scatter(y_pred, y_test, color='blue', label="Actual")
    plt.title("SVM Regression - Actual vs Predicted")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.legend()
    st.pyplot(fig)

    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_mse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_mse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)


def ridge_model(x_train, x_test, y_train, y_test):
    ridge = Ridge(alpha=1.0)

    cv_mse_scores = -cross_val_score(ridge, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = cross_val_score(ridge, x_train, y_train, scoring='r2', cv=5)

    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)

    fig = plt.figure()
    plt.scatter(y_pred, y_test, color='blue', label="Actual")
    plt.title("Ridge Regression - Actual vs Predicted")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.legend()
    st.pyplot(fig)

    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_mse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_mse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)

def add_sliders(x_train):
    feature_sliders = []
    for column in x_train.columns:
        feature_min = np.min(x_train[column])
        feature_max = np.max(x_train[column])
        feature_slider = st.slider(f"{column}", feature_min, feature_max, value=(feature_min, feature_max))
        feature_sliders.append(feature_slider)
    return feature_sliders

if st.button("Run Model"):
    match model_select:
        case "Linear":
            linear_model(x_train, x_test, y_train, y_test)
        case "Random Forest":
            random_forest(x_train, x_test, y_train, y_test)
        case "SVR":
            svr_model(x_train, x_test, y_train, y_test)
        case "Ridge":
            ridge_model(x_train, x_test, y_train, y_test)
        case "Neural Net":
            pass
        case "XGBoost":
            pass