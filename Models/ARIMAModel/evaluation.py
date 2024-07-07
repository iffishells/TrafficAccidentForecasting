# from darts.models import RegressionModel
from darts.models.forecasting.arima import ARIMA

from Models.ARIMAModel.plot import plot_visualization
import gc
import pandas as pd
import numpy as np
def load_model_sarima_model(model_name=None):
    model = ARIMA.load(model_name)
    return model

def evaluation_of_model_sarima_Model(
        model_name=None,
        ts_test=None,
        ts_train=None):
    try:

        # Load the pre-trained model
        model = load_model_sarima_model(model_name)

        # List to store evaluation results for different configurations
        evaluation_dict_list = []

         # Loop over different window sizes
        input_window = [30*2, 30 * 4, 30 * 6, 30 * 8, 30 * 10]
        for window_size in input_window:
            
            # Extract a subset of the test data based on the window size and current position
            subset_input = ts_test[:window_size]
            actual_val = ts_test[window_size:]


            # Calculate the forecast horizon
            horizon = len(actual_val)
            print(f'Evaluation of input window : {window_size} & Horizon : {horizon}')

            # Make predictions using the model
            forecasted_series = model.predict(horizon, series=subset_input)

            # Generate a filename based on the current configuration
            filename = f'input_window_{window_size}_output_window_{horizon}'

            # Inverse transform the forecasted series

            # Calculate evaluation metrics
            metrics = calculate_metrics(actual_val, forecasted_series)

            # Display the calculated metrics
            print('metrics:', metrics)

            # Store the evaluation results in a dictionary
            evaluation_dict = {
                'input_window_in_hours': window_size,
                'output_window_in_hours': horizon,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'MSE': metrics['MSE']
            }

            # Append the evaluation dictionary to the list
            evaluation_dict_list.append(evaluation_dict)

            # Uncomment the following lines if you want to plot visualizations
            plot_visualization(forecasted_series,
                                ts_train=subset_input,
                                ts_test=actual_val,
                                filename=filename)

            # Optional: Break the inner loop for testing with fewer iterations
                # Clear intermediate variables
            del subset_input, actual_val, forecasted_series, metrics

        # Create a DataFrame from the list of evaluation dictionaries
        evaluation_df = pd.DataFrame(evaluation_dict_list)

        # Perform garbage collection to free up memory
        gc.collect()
        return evaluation_df
    except Exception as e:
        print('[Error] Occurred in evaluation_of_model() : ',e)

def calculate_metrics(actual, predicted):
    try:

        # Convert inputs to numpy arrays for easier calculations
        actual = np.array(actual.values())
        predicted = np.array(predicted.values())

        # Calculate individual metrics
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        mape = np.mean(np.abs((predicted - actual) / actual)) * 100
        mse = np.mean((predicted - actual) ** 2)

        metrics = {
            "MAE": np.round(mae, 2),
            "RMSE": np.round(rmse, 2),
            "MAPE": np.round(mape, 2),
            "MSE": np.round(mse, 2),
        }
        return metrics

    except Exception as e:
        print('Error in calculate_metrics() : ',e)

