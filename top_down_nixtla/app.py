import logging
import numpy as np
from IPython.display import display
import pandas as pd
import os
import multivariate as multivariate_module
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error
from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, AutoETS, HistoricAverage,
                                  Naive, RandomWalkWithDrift, SeasonalNaive,
                                  WindowAverage, SeasonalWindowAverage)
from univariate_st import *

# Streamlit App
st.title("Time Series Forecasting")

# Navigation
page = st.sidebar.selectbox("Select Page", ["Univariate", "Multivariate"])

if page == "Univariate":
    st.header("Univariate Time Series Forecasting")

    # Upload input file
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(data.head())

        # Specify the dependent variable
        dependent_var = st.selectbox("Select the dependent variable:", [i for i in data.columns if "date"!=i])

        # Select Univariate models
        model_options = ['AutoETS', 'AutoARIMA', 'HistoricAverage', 'Naive', 'RandomWalkWithDrift', 'SeasonalNaive',
                         'WindowAverage',
                         # 'SeasonalWindowAverage'
                         ]
        selected_models = st.multiselect("Select univariate models:", model_options, default=model_options)

        if st.button("Run Forecasting Comparison"):
            if not selected_models:
                st.error("Please select at least one model.")
            else:
                try:
                    # Run forecasting comparison
                    results, best_model_name, future_forecast = run_forecasting_comparison(data, dependent_var,
                                                                                           selected_models)

                    # Display results
                    st.write("Model Comparison Results:")
                    st.write(results)
                    st.write(f"Best Model: {best_model_name}")

                    st.write("Future Forecast:")
                    st.write(future_forecast)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

elif page == "Multivariate":
    st.header("Multivariate Time Series Forecasting")
    # Load the multivariate.py script dynamically and execute its functionality
    multivariate_script_path = "multivariate.py"
    if os.path.exists(multivariate_script_path):
        # multivariate_module = load_module("multivariate", multivariate_script_path)
        multivariate_module.run_multivariate_forecasting()
    else:
        st.error(f"Multivariate script not found at path: {multivariate_script_path}")