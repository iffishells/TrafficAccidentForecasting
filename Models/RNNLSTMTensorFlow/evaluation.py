import gc
from Models.RNNLSTMTensorFlow.plot import plot_visualization
from Models.RNNLSTMTensorFlow.inference import forecast
import tensorflow as tf

import pandas as pd
import numpy as np


def load_model(model_name):
    rnn_model = RNNModel.load_from_checkpoint(model_name=model_name, best=True, map_location="cpu")
    return rnn_model


def evaluation_of_model_RnnLstm_Model(
        model_name=None,
        transformer_ob=None,
        ts_test=None,
        ts_train=None,
        time_axis = None):
    # try:

        model = tf.keras.models.load_model(model_name)


        # List to store evaluation results for different configurations
        evaluation_dict_list = []

        # Loop over different window sizes
        input_window = [30*2, 30 * 4, 30 * 6, 30 * 8, 30 * 10]
        for window_size in input_window:
            
            print(f'window_size :{window_size}')
            # Extract a subset of the test data based on the window size and current position
            subset_input = ts_test[:window_size]
            subset_input_time = time_axis[:window_size]
            
            
            actual_val = ts_test[window_size:]
            actual_val_time_axis = time_axis[window_size:]
          
            horizon = len(actual_val)

            print('Horzan :' ,horizon)
            # Calculate the forecast horizon
            print(f'Evaluation of input window : {window_size} & Horizon : {horizon}')

          

            # Transform the input data using the provided transformer object
            # subset_input = transformer_ob.transform(subset_input)

            subset_input = transformer_ob.transform(subset_input.reshape(-1,1))


            forecasted_series = forecast(model, subset_input, horizon)
            

            # Generate a filename based on the current configuration
            filename = f'input_window_{window_size}_output_window_{horizon}'

            # Inverse transform the forecasted series
            forecasted_series = transformer_ob.inverse_transform(np.array(forecasted_series).reshape(-1, 1))

            forecasted_series = list(forecasted_series.flatten())
            
            min_lenght = min( len(forecasted_series) , len(actual_val))
            print(f'Mini lenght is : {min_lenght}')
            forecasted_series = forecasted_series[:min_lenght]
            actual_val =  actual_val[:min_lenght]
            actual_val_time_axis = actual_val_time_axis[:min_lenght]
            
            # forecast_values = scaler.inverse_transform())
            print("forecasted_series : ",len(forecasted_series))
            print("actual_val : ",len(actual_val))
            print("actual_val_time_axis : ",len(actual_val_time_axis))
            
            # Calculate evaluation metrics
            metrics = calculate_metrics(actual_val,forecasted_series)

            # Display the calculated metrics
            # print('metrics:', metrics)

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
            plot_visualization(predictions=forecasted_series,
                                ts_train = transformer_ob.inverse_transform(np.array(subset_input).reshape(-1, 1)),
                                ts_train_time = subset_input_time,
                                ts_test=actual_val,
                                ts_test_time = actual_val_time_axis,
                                filename=filename)
            del subset_input, actual_val, forecasted_series, metrics
            # break

                # Optional: Break the inner loop for testing with fewer iterations
            
        # Create a DataFrame from the list of evaluation dictionaries
        evaluation_df = pd.DataFrame(evaluation_dict_list)

        # Perform garbage collection to free up memory
        gc.collect()
        return evaluation_df
    # except Exception as e:
    #     print('[Error] Occurred in evaluation_of_model() : ', e)


def calculate_metrics(actual, predicted):
    try:

        # Convert inputs to numpy arrays for easier calculations
        actual = np.array(actual)
        predicted = np.array(predicted)

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
