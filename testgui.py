
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.title('Bogalusa Reduction Efficiency Modeling')
st.sidebar.title('Options')

def correct_dtypes(df: pd.DataFrame, time_name: str):
    """
    Correct the data types of the BO RE data. 
    Transform Timestamp into datetime format and force all numerical data.
    """
    
    if type(df) != pd.DataFrame:
        raise TypeError("Data is not DataFrame!")
    
    numeric_types = (int, float)
    
    for col in df:
        if col == time_name:
            df[col] = pd.to_datetime(df[col])
        elif df[col].dtype not in numeric_types:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def gen_cleaning(df: pd.DataFrame, time_name):
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
    # df = df[df['RE test'] >= 60]
    df.set_index(time_name, inplace=True)
    return df

@st.cache_data()
def load(uploaded_file):
    """_summary_

    Returns:
        _type_: _description_
    """
    if uploaded_file[-4:] == ".csv":
        data = pd.read_csv(uploaded_file)
    elif uploaded_file[-5:] == ".xlsx":
        data = pd.read_excel(uploaded_file)
    else:
        raise TypeError("Invalid file type!")

    if y_name not in data.columns:
        raise ValueError("Target variable not in columns!")
    if time_name not in data.columns:
        raise ValueError("Time column name not in columns!")
    data = correct_dtypes(data, time_name)
    data = gen_cleaning(data, time_name)
    return data
    

@st.cache_data()
def feature_selection(data, x, y):
    lasso = Lasso(alpha = 0.3)
    lasso.fit(x, y)
    feature_importances = abs(lasso.coef_)
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = x.columns[sorted_indices]
    selected_features = sorted_features[feature_importances[sorted_indices] >= 0.1]
    selected_df = data[selected_features.to_list()]
    return selected_df

# uploaded_file = st.sidebar.file_uploader("Choose a xlsx or csv file.")
# y_name = st.sidebar.text_input(label="Name of y variable:")
# time_name = st.sidebar.text_input(label="Name of time variable:")

# inputs_added = (uploaded_file is not None) & (y_name != "") & (time_name != "")




y_name = "RE test"
time_name = "Timestamp"
    
df = load("BORE_v3.csv")
x = df.drop(y_name, axis = 1)
y = df[y_name]


st.sidebar.subheader("Toggle display")
if st.sidebar.checkbox("Display data", False):
    st.subheader("Bogalusa Reduction Efficiency Dataset")
    st.write(df)



selected_df = feature_selection(df, x, y)
x_train, x_test, y_train, y_test = train_test_split(selected_df, y, test_size=0.3, random_state=24)


if st.sidebar.checkbox("Display LASSO-selected features", False):
    st.subheader("Selected BO-RE Features:")
    st.write(selected_df)

if st.sidebar.checkbox("Display EDA for Reduction Efficiency", False):
    st.subheader("EDA for Reduction Efficiency")
    fig1 = px.line(x=x.index, y=y)
    fig1.update_layout(title="Reduction Efficiency vs Time",
                    xaxis_title="Date",
                    yaxis_title="Reduction Efficiency (%)")
    st.plotly_chart(fig1, theme=None)
    
    fig2 = px.histogram(y)
    st.plotly_chart(fig2, theme=None)
    


st.sidebar.subheader("Choose model:")
model_select = st.sidebar.selectbox("Model", ("Linear", "Random Forest", "SVR", "Ridge", "Neural Net", "XGBoost"))

st.sidebar.subheader("Predict reduction efficiency for:")
st.sidebar.write("Features are listed in order of importance (measured by LASSO).")
st.sidebar.write("The topmost features are the most important.")
st.sidebar.write("You can either drag the sliders or type in your desired value.")

last_row = df.iloc[-1].to_dict()

with st.sidebar:
    for i in range(len(x_train.columns)):
        
        column = x_train.columns[i]
        feature_min = float(np.min(x_train[column]) - 5.0)
        feature_max = float(np.max(x_train[column]) + 5.0)
        # print(st.session_state)
        col_value = st.slider(f"{str(column)}",
                        value = st.session_state[f"text{i}"] if f"text{i}" in st.session_state else last_row[column],
                        min_value=feature_min, 
                        max_value=feature_max, 
                        key=f"col{i}"
                        )
        textbox = st.number_input(f"Desired Value", value=float(st.session_state[f"col{i}"]) if f"col{i}" in st.session_state else float(col_value),
                                min_value=feature_min,
                                max_value=feature_max,
                                step=5.0,
                                key=f"text{i}")


def linear_predict(x_train, x_test, y_train):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    st.write(f"**Predicted reduction efficiency based on slider values: {round(100 if y_pred[-1] > 100 else y_pred[-1], 2)}%**")


def linear_model(x_train, x_test, y_train, y_test):
    
    lr = LinearRegression()

    cv_rmse_scores = -cross_val_score(lr, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = cross_val_score(lr, x_train, y_train, scoring='r2', cv=5)

    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)


    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_rmse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_rmse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)

    # fig = plt.figure()
    # plt.scatter(y_pred, y_test, color='blue', label="Actual")
    # plt.title("Linear Regression - Actual vs Predicted")
    # plt.xlabel("Predicted values")
    # plt.ylabel("Actual values")
    # plt.legend()
    # st.pyplot(fig)
    
    fig_min = float(min(y_pred.min(), y_test.min()) - 5)
    fig_max = float(max(y_pred.max(), y_test.max()) + 5)
    fig = make_subplots()
    fig.update_layout(title= "Linear Regression - Actual vs Predicted",
                        xaxis_title="Predicted values", 
                        yaxis_title="Actual values")
    fig.add_trace(go.Scatter(x=y_pred, 
                            y=y_test,
                            showlegend=False, 
                            mode="markers"))
    fig.add_shape(type="line", 
                x0=fig_min - 10, 
                y0=fig_min - 10, 
                x1=fig_max + 10, 
                y1=fig_max + 10,
                line=dict(color="#d62728",width=3))
    fig.add_trace(go.Scatter(
        x=list(range(round(fig_min) - 1, round(fig_max) + 1)), 
        y=list(range(round(fig_min) - 1, round(fig_max) + 1)),  
        mode="lines", 
        name="",
        showlegend=False, 
        text="The closer the dots are to this line, the better the predictions are.",
        opacity=0))
    
    fig.update_xaxes(range=[fig_min, fig_max])
    fig.update_yaxes(range=[fig_min, fig_max])
    
    st.plotly_chart(fig, theme=None)
    


def rf_predict(x_train, x_test, y_train):
    rf_regressor = RandomForestRegressor(n_estimators=350)
    rf_regressor.fit(x_train, y_train)
    y_pred = rf_regressor.predict(x_test)
    st.write(f"**Predicted reduction efficiency based on slider values: {round(100 if y_pred[-1] > 100 else y_pred[-1], 2)}%**")

def random_forest(x_train, x_test, y_train, y_test):
    rf_regressor = RandomForestRegressor(n_estimators=350)

    cv_mse_scores = -cross_val_score(rf_regressor, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = 1 - (cv_mse_scores / np.var(y_train))

    rf_regressor.fit(x_train, y_train)
    y_pred = rf_regressor.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)

    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_mse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_mse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)

    fig_min = float(min(y_pred.min(), y_test.min()) - 5)
    fig_max = float(max(y_pred.max(), y_test.max()) + 5)
    fig = make_subplots()
    fig.update_layout(title= "Random Forest Regression - Actual vs Predicted",
                        xaxis_title="Predicted values", 
                        yaxis_title="Actual values")
    fig.add_trace(go.Scatter(x=y_pred, 
                            y=y_test,
                            showlegend=False, 
                            mode="markers"))
    fig.add_shape(type="line", 
                x0=fig_min - 10, 
                y0=fig_min - 10, 
                x1=fig_max + 10, 
                y1=fig_max + 10,
                line=dict(color="#d62728",width=3))
    fig.add_trace(go.Scatter(
        x=list(range(round(fig_min) - 1, round(fig_max) + 1)), 
        y=list(range(round(fig_min) - 1, round(fig_max) + 1)),  
        mode="lines", 
        name="",
        showlegend=False, 
        text="The closer the dots are to this line, the better the predictions are.",
        opacity=0))
    
    fig.update_xaxes(range=[fig_min, fig_max])
    fig.update_yaxes(range=[fig_min, fig_max])
    
    st.plotly_chart(fig, theme=None)


def svr_predict(x_train, x_test, y_train):
    svr = SVR(kernel='rbf')
    svr.fit(x_train, y_train)
    y_pred = svr.predict(x_test)
    st.write(f"**Predicted reduction efficiency based on slider values: {round(100 if y_pred[-1] > 100 else y_pred[-1], 2)}%**")


def svr_model(x_train, x_test, y_train, y_test):
    svr = SVR(kernel='rbf')

    cv_mse_scores = -cross_val_score(svr, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = cross_val_score(svr, x_train, y_train, scoring='r2', cv=5)

    svr.fit(x_train, y_train)
    y_pred = svr.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)

    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_mse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_mse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)

    fig_min = float(min(y_pred.min(), y_test.min()) - 5)
    fig_max = float(max(y_pred.max(), y_test.max()) + 5)
    fig = make_subplots()
    fig.update_layout(title= "SVM Regression - Actual vs Predicted",
                        xaxis_title="Predicted values", 
                        yaxis_title="Actual values")
    fig.add_trace(go.Scatter(x=y_pred, 
                            y=y_test,
                            showlegend=False, 
                            mode="markers"))
    fig.add_shape(type="line", 
                x0=fig_min - 10, 
                y0=fig_min - 10, 
                x1=fig_max + 10, 
                y1=fig_max + 10,
                line=dict(color="#d62728",width=3))
    fig.add_trace(go.Scatter(
        x=list(range(round(fig_min) - 1, round(fig_max) + 1)), 
        y=list(range(round(fig_min) - 1, round(fig_max) + 1)),  
        mode="lines", 
        name="",
        showlegend=False, 
        text="The closer the dots are to this line, the better the predictions are.",
        opacity=0))
    
    fig.update_xaxes(range=[fig_min, fig_max])
    fig.update_yaxes(range=[fig_min, fig_max])
    
    st.plotly_chart(fig, theme=None)


def ridge_predict(x_train, x_test, y_train):
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    st.write(f"**Predicted reduction efficiency based on slider values: {round(100 if y_pred[-1] > 100 else y_pred[-1], 2)}%**")


def ridge_model(x_train, x_test, y_train, y_test):
    ridge = Ridge(alpha=1.0)

    cv_mse_scores = -cross_val_score(ridge, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
    cv_r2_scores = cross_val_score(ridge, x_train, y_train, scoring='r2', cv=5)

    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_r2 = r2_score(y_test, y_pred)

    st.write("Cross-Validation RMSE Scores:")
    st.write(np.sqrt(cv_mse_scores)[np.newaxis])
    st.write("Average RMSE: ", np.mean(np.sqrt(cv_mse_scores)))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))
    st.write("\nTest RMSE: ", test_rmse)
    st.write("Test R^2: ", test_r2)

    fig_min = float(min(y_pred.min(), y_test.min()) - 5)
    fig_max = float(max(y_pred.max(), y_test.max()) + 5)
    fig = make_subplots()
    fig.update_layout(title= "Ridge Regression - Actual vs Predicted",
                        xaxis_title="Predicted values", 
                        yaxis_title="Actual values")
    fig.add_trace(go.Scatter(x=y_pred, 
                            y=y_test,
                            showlegend=False, 
                            mode="markers"))
    fig.add_shape(type="line", 
                x0=fig_min - 10, 
                y0=fig_min - 10, 
                x1=fig_max + 10, 
                y1=fig_max + 10,
                line=dict(color="#d62728",width=3))
    fig.add_trace(go.Scatter(
        x=list(range(round(fig_min) - 1, round(fig_max) + 1)), 
        y=list(range(round(fig_min) - 1, round(fig_max) + 1)),  
        mode="lines", 
        name="",
        showlegend=False, 
        text="The closer the dots are to this line, the better the predictions are.",
        opacity=0))
    
    fig.update_xaxes(range=[fig_min, fig_max])
    fig.update_yaxes(range=[fig_min, fig_max])
    
    st.plotly_chart(fig, theme=None)


def neural_predict(x_train, x_test, y_train):
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

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).numpy()

    st.write(f"**Predicted reduction efficiency based on slider values: {round(100 if y_pred[-1][0] > 100 else y_pred[-1][0], 2)}%**")


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

        # st.write('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss))

        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(x_val)
                predicted_values.append(y_pred)

                # fig1 = plt.figure()
                # plt.scatter(y_val, y_pred)
                # plt.xlabel('Actual y')
                # plt.ylabel('Predicted y')
                # plt.title('Predicted vs Actual (Epoch %d)' % (epoch+1))
                # st.pyplot(fig1)
                # fig1.clf()

                # fig1 = px.scatter(x=y_pred.numpy().flatten(), y=y_val.numpy().flatten())
                # fig1.update_layout(title=f"Predicted vs Actual (Epoch {epoch + 1})",
                #                    xaxis_title="Predicted y",
                #                    yaxis_title="Actual y")
                # st.plotly_chart(fig1, theme=None)
                
                fig_min = float(min(y_pred.numpy().min(), y_val.numpy().min()) - 5)
                fig_max = float(max(y_pred.numpy().max(), y_val.numpy().max()) + 5)
                fig = make_subplots()
                fig.update_layout(title= f"Predicted vs Actual (Epoch {epoch + 1})",
                                    xaxis_title="Predicted y", 
                                    yaxis_title="Actual y")
                fig.add_trace(go.Scatter(x=y_pred.numpy().flatten(), 
                                        y=y_val.numpy().flatten(),
                                        showlegend=False, 
                                        mode="markers"))
                fig.add_shape(type="line", 
                            x0=fig_min - 10, 
                            y0=fig_min - 10, 
                            x1=fig_max + 10, 
                            y1=fig_max + 10,
                            line=dict(color="#d62728",width=3))
                fig.add_trace(go.Scatter(
                    x=list(range(round(fig_min) - 1, round(fig_max) + 1)), 
                    y=list(range(round(fig_min) - 1, round(fig_max) + 1)),  
                    mode="lines", 
                    name="",
                    showlegend=False, 
                    text="The closer the dots are to this line, the better the predictions are.",
                    opacity=0))
                
                fig.update_xaxes(range=[fig_min, fig_max])
                fig.update_yaxes(range=[fig_min, fig_max])
                
                st.plotly_chart(fig, theme=None)
                
                

    


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

    # fig2 = plt.figure()
    # plt.plot(range(num_epochs), epoch_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Epoch Loss')
    # st.pyplot(fig2)
    
    fig2 = px.line(x=list(range(num_epochs)), y=epoch_losses)
    fig2.update_layout(title= "Epoch Loss",
                        xaxis_title="Epoch", 
                        yaxis_title="Loss")
    st.plotly_chart(fig2, theme=None)
    


def xgboost_predict(x_train, x_test, y_train):
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    st.write(f"**Predicted reduction efficiency based on slider values: {round(100 if y_pred[-1] > 100 else y_pred[-1], 2)}%**")

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


    st.write("Cross-Validation MSE Scores:")
    st.write(cv_mse_scores[np.newaxis])
    st.write("Average MSE: ", np.mean(cv_mse_scores))
    st.write("\nCross-Validation R^2 Scores:")
    st.write(cv_r2_scores[np.newaxis])
    st.write("Average R^2: ", np.mean(cv_r2_scores))

    
    fig_min = float(min(y_pred.min(), y_test.min()) - 5)
    fig_max = float(max(y_pred.max(), y_test.max()) + 5)
    fig = make_subplots()
    fig.update_layout(title= "XGBoost Regression - Actual vs Predicted",
                        xaxis_title="Predicted values", 
                        yaxis_title="Actual values")
    fig.add_trace(go.Scatter(x=y_pred, 
                            y=y_test,
                            showlegend=False, 
                            mode="markers"))
    fig.add_shape(type="line", 
                x0=fig_min - 10, 
                y0=fig_min - 10, 
                x1=fig_max + 10, 
                y1=fig_max + 10,
                line=dict(color="#d62728",width=3))
    fig.add_trace(go.Scatter(
        x=list(range(round(fig_min) - 1, round(fig_max) + 1)), 
        y=list(range(round(fig_min) - 1, round(fig_max) + 1)),  
        mode="lines", 
        name="",
        showlegend=False, 
        text="The closer the dots are to this line, the better the predictions are.",
        opacity=0))
    
    fig.update_xaxes(range=[fig_min, fig_max])
    fig.update_yaxes(range=[fig_min, fig_max])
    
    st.plotly_chart(fig, theme=None)


slider_values = {}
def retrieve_slider_values(): 
    for i in range(len(x_train.columns)): 
        column = x_train.columns[i] 
        slider_values[column] = st.session_state[f"col{i}"]
        # st.write(slider_values[column])

run_check = st.checkbox("Run Model")
predict_check = st.checkbox("Predict")
retrieve_slider_values()
new_row = pd.DataFrame(slider_values, index=["Current Slider Values"])
st.write(new_row)


if run_check:
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


if predict_check: 
    
    x_test2 = pd.concat([x_test, new_row], ignore_index=True)
    match model_select:
        case "Linear":
            linear_predict(x_train, x_test2, y_train)
        case "Random Forest":
            rf_predict(x_train, x_test2, y_train)
        case "SVR":
            svr_predict(x_train, x_test2, y_train)
        case "Ridge":
            ridge_predict(x_train, x_test2, y_train)
        case "Neural Net":
            neural_predict(x_train, x_test2, y_train)
        case "XGBoost":
            xgboost_predict(x_train, x_test2, y_train)