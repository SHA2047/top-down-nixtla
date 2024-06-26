import logging
import numpy as np
from IPython.display import display
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error
from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, AutoETS, HistoricAverage,
                                  Naive, RandomWalkWithDrift, SeasonalNaive,
                                  WindowAverage, SeasonalWindowAverage)

# Configure logging
logging.basicConfig(filename='run_time_of_run.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from rollingmodelevaluator import *
# TimeSeriesModel Base Class
class TimeSeriesModel:
    def __init__(self, name):
        self.name = name
        self.model = None

    def fit(self, train_data):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, forecast_horizon):
        raise NotImplementedError("Subclasses should implement this method.")

# Univariate Models Classes (using nixtla statsforecast)
class AutoETSModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('AutoETS')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1  # Add unique_id for the single time series
        self.model = StatsForecast(models=[AutoETS(season_length=12)], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df[self.name]

class AutoARIMAModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('AutoARIMA')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1  # Add unique_id for the single time series
        self.model = StatsForecast(models=[AutoARIMA()], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df[self.name]

class HistoricAverageModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('HistoricAverage')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1
        self.model = StatsForecast(models=[HistoricAverage()], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df[self.name]

class NaiveModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('Naive')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1
        self.model = StatsForecast(models=[Naive()], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df[self.name]

class RandomWalkWithDriftModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('RandomWalkWithDrift')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1
        self.model = StatsForecast(models=[RandomWalkWithDrift()], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df['RWD']

class SeasonalNaiveModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('SeasonalNaive')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1
        self.model = StatsForecast(models=[SeasonalNaive(season_length=12)], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df[self.name]

class WindowAverageModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('WindowAverage')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1
        self.model = StatsForecast(models=[WindowAverage(window_size=12)], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df[self.name]

class SeasonalWindowAverageModel(TimeSeriesModel):
    def __init__(self):
        super().__init__('SeasonalWindowAverage')

    def fit(self, train_data):
        train_data.loc[:, 'unique_id'] = 1
        self.model = StatsForecast(models=[SeasonalWindowAverage(season_length=12, window_size=12)], freq='M')
        self.model.fit(df=train_data)

    def predict(self, forecast_horizon):
        forecast_df = self.model.predict(h=forecast_horizon)
        return forecast_df[self.name]

# ModelComparison Class
class ModelComparison:
    def __init__(self, models, data, window_size=6):
        self.models = models
        self.data = data
        self.window_size = window_size

    def run(self):
        results = {}
        for model in self.models:
            evaluator = RollingModelEvaluator(model, self.data, self.window_size)
            evaluation_results, rolling_accuracies = evaluator.evaluate()
            mape_mean = evaluation_results['mape'].mean()
            results[model.name] = mape_mean
            logging.info(f"Model: {model.name}, Average MAPE: {mape_mean:.4f}")
            st.write(f"Model: {model.name}")
            st.dataframe(rolling_accuracies)
            st.table(evaluation_results)
        return results

# Function to run the full process
def run_forecasting_comparison(data, dependent_var, selected_models):
    logging.info("Starting the forecasting comparison process")

    # Convert date column to datetime and rename columns to match StatsForecast expectations
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    data.rename(columns={'date': 'ds', dependent_var: 'y'}, inplace=True)

    # Splitting the data into training (non-missing values) and testing (missing values)
    train_data = data[data['y'].notna()]
    test_data = data[data['y'].isna()]

    logging.info(f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")

    # Initialize models
    model_classes = {
        'AutoETS': AutoETSModel, 'AutoARIMA': AutoARIMAModel, 'HistoricAverage': HistoricAverageModel,
        'Naive': NaiveModel, 'RandomWalkWithDrift': RandomWalkWithDriftModel, 'SeasonalNaive': SeasonalNaiveModel,
        'WindowAverage': WindowAverageModel, 'SeasonalWindowAverage': SeasonalWindowAverageModel
    }

    models = [model_classes[model]() for model in selected_models]

    # Run model comparison

    comparison = ModelComparison(models=models, data=data, window_size=6)
    results = comparison.run()

    # Select the best model based on MAPE
    best_model_name = min(results, key=results.get)
    best_model = next(model for model in models if model.name == best_model_name)

    logging.info(f"Best model selected: {best_model_name}")

    # Fit the best model on the entire data with non-missing values
    best_model.fit(train_data)

    # Forecast starting from the first missing value date
    forecast_start_date = data[data['y'].isna()]['ds'].iloc[0]
    forecast_horizon = len(data) - len(train_data)
    future_dates = pd.date_range(start=forecast_start_date, periods=forecast_horizon, freq='M')
    future_forecast = best_model.predict(forecast_horizon)

    # Return results and the final forecast
    logging.info("Forecasting process completed")
    return results, best_model_name, pd.DataFrame({'date': future_dates, 'forecast': future_forecast})


