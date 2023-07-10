
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
import modeling 

st.title('Bogalusa Reduction Efficiency Model')
st.sidebar.title('Options')

@st.cache_data(persist= True)
def load():
    data = pd.read_csv("BORE_v3.csv")
    data = modeling.correct_dtypes(data)
    data = modeling.gen_cleaning(data)
    return data
df = load()
x = df.drop("RE test", axis = 1)
y = df["RE test"]

st.sidebar.subheader("Toggle display")
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
x_train, x_test, y_train, y_test = train_test_split(selected_df, y, test_size=0.3, random_state=24)


if st.sidebar.checkbox("Display LASSO-selected features", False):
    st.subheader("Selected BO-RE Features:")
    st.write(selected_df)

st.sidebar.subheader("Choose model:")
model_select = st.sidebar.selectbox("Model", ("Linear", "Random Forest", "SVR", "Ridge", "Neural Net", "XGBoost"))

st.sidebar.subheader("Tune model:")

last_row = df.iloc[-1].to_dict()
with st.sidebar:
    for i in range(len(x_train.columns)):
        column = x_train.columns[i]
        feature_min = float(np.min(x_train[column]) - 5.0)
        feature_max = float(np.max(x_train[column]) + 5.0)
        value = st.slider(f"{str(column)}",
                          value = exec("st.session_state.col" + str(i)) if f"col{i}" in st.session_state else last_row[column],
                          min_value=feature_min, 
                          max_value=feature_max, 
                          key=f"col{i}"
                          )


@st.cache_data(persist=True)


def linear_predict(x_train, x_test, y_train):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred[-1]

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

def neural_net(x_train, x_test, y_train, y_test):
    epoch_losses = []
    predicted_values = []

    x_train_array = x_train.values
    y_train_array = y_train.values
    x_test_array = x_test.values

    x_train_tensor = torch.from_numpy(x_train_array).float()
    y_train_tensor = torch.from_numpy(y_train_array).float()
    x_test_tensor = torch.from_numpy(x_test_array).float()

    x_train, x_val, y_train, y_val = train_test_split(x_train_tensor, y_train_tensor, test_size=0.2, random_state=43)

    class NeuralNet(nn.Module):
        def __init__(self, input_size):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = NeuralNet(input_size=x_train_tensor.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 100
    batch_size = 32


    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            inputs = x_train[i:i+batch_size]
            labels = y_train[i:i+batch_size]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_losses.append(running_loss)

        st.write('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss))

        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(x_val)
                predicted_values.append(y_pred)

                fig1 = plt.figure()
                plt.scatter(y_val, y_pred)
                plt.xlabel('Actual y')
                plt.ylabel('Predicted y')
                plt.title('Predicted vs Actual (Epoch %d)' % (epoch+1))
                st.pyplot(fig1)
                fig1.clf()

    fig2 = plt.figure()
    plt.plot(range(num_epochs), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss')
    st.pyplot(fig2)


    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write("Test RMSE:", rmse)


    model.eval()
    with torch.no_grad():
        y_pred = model(x_val)
        val_loss = criterion(y_pred, y_val)
        val_rmse = torch.sqrt(val_loss).item()
        st.write("Validation RMSE:", val_rmse)


    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write("Test RMSE:", rmse)

def xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBRegressor()

    cv_mse_scores = -cross_val_score(xgb_model, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = cross_val_score(xgb_model, x_train, y_train, scoring='r2', cv=5)

    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write("RMSE:", rmse)

    r2 = r2_score(y_test, y_pred)
    st.write("R2 score:", r2)

    fig = plt.figure()
    plt.scatter(y_pred, y_test, color='blue', label="Actual")
    plt.title("XGBoost Regression - Actual vs Predicted")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.legend()
    st.pyplot(fig)

    st.write("Cross-Validation MSE Scores:")
    st.write(cv_mse_scores[np.newaxis])
    st.write("Average MSE: ", np.mean(cv_mse_scores))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))


slider_values = {}
def retrieve_slider_values(): 
    for i in range(len(x_train.columns)): 
        column = x_train.columns[i] 
        slider_values[column] = st.session_state[f"col{i}"]
        # st.write(slider_values[column])

run_button = st.button("Run Model")
if run_button:
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
            neural_net(x_train, x_test, y_train, y_test)
        case "XGBoost":
            xgboost(x_train, x_test, y_train, y_test)

predict_button = st.button("Predict")
if predict_button: 
    retrieve_slider_values()
    new_row = pd.DataFrame(slider_values, index=[0])
    x_test2 = pd.concat([x_test, new_row], ignore_index=True)
    match model_select:
        case "Linear":
            y_pred = linear_predict(x_train, x_test2, y_train)
            st.write(f"y_pred based off slider values: {y_pred}")