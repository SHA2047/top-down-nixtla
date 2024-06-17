import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def load_data(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        return pd.read_csv(file)
    elif file_extension == 'xlsx':
        return pd.read_excel(file)
    elif file_extension == 'parquet':
        return pd.read_parquet(file)
    else:
        st.error('Unsupported file type')
        return None


def preprocess_data(data):
    data = data.dropna()
    return data


def train_model(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, mae


def forecast(model, data, target_column):
    future_data = data.drop(columns=[target_column])
    forecasted_values = model.predict(future_data)
    return forecasted_values


def run_multivariate_forecasting():
    st.write("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "parquet"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data Preview")
            st.dataframe(data.head())

            target_column = st.selectbox("Select the target column for forecasting", data.columns)

            if st.button("Train and Forecast"):
                data = preprocess_data(data)
                model, mae = train_model(data, target_column)
                st.write(f"Model trained. Mean Absolute Error: {mae:.4f}")

                forecasted_values = forecast(model, data, target_column)
                forecasted_df = pd.DataFrame({target_column: forecasted_values})

                st.write("Forecasted Values")
                st.dataframe(forecasted_df)
