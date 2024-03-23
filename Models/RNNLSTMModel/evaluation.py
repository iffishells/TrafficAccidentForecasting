import gc
from darts.models import RNNModel
from Models.RNNLSTMModel.plot import plot_visualization
from Models.RNNLSTMModel.preprocessing import inverse_transformed
import pandas as pd
import numpy as np


def load_model(model_name):
    rnn_model = RNNModel.load_from_checkpoint(model_name=model_name, best=True, map_location="cpu")
    return rnn_model


def evaluation_of_model_RnnLstm_Model(
        model_name=None,
        transformer_ob=None,
        ts_test=None,
        ts_train=None):
    try:

        # Load the pre-trained model
        model = load_model(model_name)

        # List to store evaluation results for different configurations
        evaluation_dict_list = []

        # Loop over different window sizes
        input_window = [30, 30 * 2, 30 * 3, 30 * 4, 30 * 5]
        for window_size in input_window:
            print(f'window_size :{window_size}')
            # Extract a subset of the test data based on the window size and current position
            subset_input = ts_test[:window_size]
            actual_val = ts_test[window_size:]
            horizon = len(actual_val)
            # Calculate the forecast horizon
            print(f'Evaluation of input window : {window_size} & Horizon : {horizon}')

          

            # Transform the input data using the provided transformer object
            subset_input = transformer_ob.transform(subset_input)

            # Make predictions using the model
            forecasted_series = model.predict(horizon, series=subset_input)

            # Generate a filename based on the current configuration
            filename = f'input_window_{window_size}_output_window_{horizon}'

            # Inverse transform the forecasted series
            forecasted_series = inverse_transformed(transformer_ob, forecasted_series)

            # Calculate evaluation metrics
            metrics = calculate_metrics(actual_val, forecasted_series)

            # Display the calculated metrics
            print('metrics:', metrics)

            # Store the evaluation results in a dictionary
            evaluation_dict = {
                    'input_window_in_hours' : window_size,
                    'output_window_in_hours': horizon,
                    'MAE'                   : metrics['MAE'],
                    'RMSE'                  : metrics['RMSE'],
                    'MAPE'                  : metrics['MAPE'],
                    'MSE'                   : metrics['MSE']
            }

            # Append the evaluation dictionary to the list
            evaluation_dict_list.append(evaluation_dict)

            # Uncomment the following lines if you want to plot visualizations
            plot_visualization(forecasted_series,
                                ts_train= inverse_transformed(transformer_ob, subset_input),
                                ts_test=actual_val,
                                filename=filename)
            del subset_input, actual_val, forecasted_series, metrics

                # Optional: Break the inner loop for testing with fewer iterations
                # break

        # Create a DataFrame from the list of evaluation dictionaries
        evaluation_df = pd.DataFrame(evaluation_dict_list)

        # Perform garbage collection to free up memory
        gc.collect()
        return evaluation_df
    except Exception as e:
        print('[Error] Occurred in evaluation_of_model() : ', e)


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
                "MAE" : np.round(mae, 2),
                "RMSE": np.round(rmse, 2),
                "MAPE": np.round(mape, 2),
                "MSE" : np.round(mse, 2),
        }
        return metrics

    except Exception as e:
        print('Error in calculate_metrics() : ', e)

# def evaluation_of_model_RnnLstm_Model(
#         model_name=None,
#         transformer_ob=None,
#         ts_test=None,
#         ts_train=None):
#     try:
#
#         # Load the pre-trained model
#         model = load_model(model_name)
#
#         # List to store evaluation results for different configurations
#         evaluation_dict_list = []
#
#         # Loop over different window sizes
#         for window_size in [24, 24*2, 24*3, 24*4]:  # 1 day, 2 days, 3 days, 4 days
#
#             # Iterate over different starting positions in the test set
#             for i in range(10):
#                 # Extract a subset of the test data based on the window size and current position
#                 subset_input = ts_test[i: i + window_size]
#                 actual_val = ts_test[i + window_size:]
#
#                 # Calculate the forecast horizon
#                 horizon = len(actual_val)
#
#                 # Transform the input data using the provided transformer object
#                 subset_input = transformer_ob.transform(subset_input)
#
#                 # Make predictions using the model
#                 forecasted_series = model.predict(horizon, series=subset_input)
#
#                 # Generate a filename based on the current configuration
#                 filename = f'slide_{i}_input_window_{window_size}_output_window_{horizon}'
#
#                 # Inverse transform the forecasted series
#                 forecasted_series = inverse_transformed(transformer_ob, forecasted_series)
#
#                 # Calculate evaluation metrics
#                 metrics = calculate_metrics(actual_val, forecasted_series)
#
#                 # Display the calculated metrics
#                 print('metrics:', metrics)
#
#                 # Store the evaluation results in a dictionary
#                 evaluation_dict = {
#                     'slide': i,
#                     'input_window_in_hours': window_size,
#                     'output_window_in_hours': horizon,
#                     'MAE': metrics['MAE'],
#                     'RMSE': metrics['RMSE'],
#                     'MAPE': metrics['MAPE'],
#                     'MSE': metrics['MSE']
#                 }
#
#                 # Append the evaluation dictionary to the list
#                 evaluation_dict_list.append(evaluation_dict)
#
#                 # Uncomment the following lines if you want to plot visualizations
#                 plot_visualization(forecasted_series,
#                                    ts_train=ts_train,
#                                    ts_test=actual_val,
#                                    filename=filename)
#
#                 # Optional: Break the inner loop for testing with fewer iterations
#                 # break
#
#         # Create a DataFrame from the list of evaluation dictionaries
#         evaluation_df = pd.DataFrame(evaluation_dict_list)
#
#         # Perform garbage collection to free up memory
#         gc.collect()
#         return evaluation_df
#     except Exception as e:
#         print('[Error] Occured in evaluation_of_model() : ',e)
