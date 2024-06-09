import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


class RollingModelEvaluator:
    def __init__(self, model, test_data, window_size):
        """
        Initializes the RollingModelEvaluator with a model, test data, and window size.

        :param model: A pre-trained time series model
        :param test_data: DataFrame with columns ['date', 'actual'] and any other features required by the model
        :param window_size: Number of observations to include in each rolling window
        """
        self.model = model
        self.test_data = test_data
        self.window_size = window_size

    def evaluate(self):
        """
        Evaluates the model over a rolling window.

        :return: DataFrame with columns ['start_date', 'end_date', 'mae']
        """
        results = []
        dates = self.test_data['date']

        for start in range(len(dates) - self.window_size + 1):
            end = start + self.window_size
            window_data = self.test_data.iloc[start:end]
            actuals = window_data['actual'].values
            features = window_data.drop(columns=['date', 'actual']).values

            # Predict using the model
            predictions = self.model.predict(features)

            # Calculate MAE for the window
            mae = mean_absolute_error(actuals, predictions)
            results.append({
                'start_date': dates.iloc[start],
                'end_date': dates.iloc[end - 1],
                'mae': mae
            })

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Dummy example model and data
    class DummyModel:
        def predict(self, X):
            return np.mean(X, axis=1)  # Dummy prediction logic


    # Create dummy test data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'date': dates,
        'actual': np.random.randn(100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })

    # Initialize and evaluate
    dummy_model = DummyModel()
    evaluator = RollingModelEvaluator(model=dummy_model, test_data=test_data, window_size=20)
    results = evaluator.evaluate()

    # Display results
    print(results)
