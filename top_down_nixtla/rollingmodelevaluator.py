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
from univariate_st import *

# RollingModelEvaluator Class
class RollingModelEvaluator:
    def __init__(self, model, data, window_size=6):
        self.model = model
        self.data = data
        self.window_size = window_size

    def evaluate(self):
        results = []
        rolling_accuracies = pd.DataFrame()
        last_train_date = self.data[self.data['y'].notna()]['ds'].max()
        for i in range(self.window_size):
            i = 6 - i
            train_end_date = last_train_date - pd.DateOffset(months=i)
            train_data = self.data[self.data['ds'] <= train_end_date].copy()
            test_data = self.data[(self.data['ds'] > train_end_date) & (self.data['y'].notna())].copy()

            if not test_data.empty:
                self.model.fit(train_data)
                predictions = self.model.predict(len(test_data))
                print("length:")
                print(len(predictions))
                mape = mean_absolute_percentage_error(test_data['y'].values, predictions)
                results.append({
                    'train_end_date': train_end_date,
                    'test_end_date': test_data['ds'].max(),
                    'mape': mape
                })

                logging.info(f"Model: {self.model.name}, Train End Date: {train_end_date}, "
                             f"MAPE: {mape:.4f}")

            predictions = predictions.values
            print("Here:")
            display(pd.DataFrame(predictions, index=test_data['ds'].tolist(), columns=['y']))
            display(test_data.set_index('ds')[['y']])
            accuracies = np.subtract(1, np.abs(1 - pd.DataFrame(predictions, index=test_data['ds'].tolist(), columns=['y']) / test_data.set_index('ds')[['y']]))
            accuracies = accuracies.rename(columns={'y': train_end_date}).T
            rolling_accuracies = pd.concat([rolling_accuracies, accuracies])

        return (pd.DataFrame(results), rolling_accuracies)