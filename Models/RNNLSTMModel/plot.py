
import os
import sys
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def train_test_predicted_plot(df_train, df_test, x_feature, y_feature, predicted, model_name, filename):
    """
    Plots the training data, actual values, and forecasted values using Plotly.

    Args:
        train (pd.Series): The training data.
        test (pd.Series): The actual values.
        predicted (pd.Series): The forecasted values.
        model_name (str): The name of the forecasting model.

    Returns:
        None
    """

    # Create a subplot with two rows and one column
    try:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_train[x_feature],
                y=df_train[y_feature],
                name='Input Series',
                mode='lines+markers'
            ))

        # Add a trace for actual values
        fig.add_trace(
            go.Scatter(
                x=df_test[x_feature],
                y=df_test[y_feature],
                name='Actual Values',
                mode='lines+markers'
            )
        )

        # Add a trace for forecasted values
        fig.add_trace(
            go.Scatter(
                x=df_test[x_feature],
                y=predicted[y_feature],
                name=f'Rnn-Lstm Prediction',
                mode='lines+markers'
            )
        )

        # Update xaxis properties
        fig.update_xaxes(title_text='Time')

        # Update yaxis properties
        fig.update_yaxes(title_text=y_feature)

        # Update title and height
        title = f'Forecasting using {model_name}\ninput window size :{df_train.shape[0]}\n Horizon : {df_test.shape[0]}'

        fig.update_layout(
            title=title,
            height=500,
            width=1500,
            legend=dict(x=0, y=1, traceorder='normal', orientation='h')
        )

        # Save the plot as an HTML file
        # fig.show()
        parent_path = os.path.join('..','Plots','RNNLSTMModel','Results')
        os.makedirs(parent_path,exist_ok=True)
        fig.write_html(f'{parent_path}/forecasting_using_{model_name}_combination_{filename}'+'.html')
        fig.write_image(f'{parent_path}/forecasting_using_{model_name}_combination_of_{filename}' + '.png')
        fig.show()
    except Exception as e:
        print("Error ", e)

def plot_visualization(predictions=None,
                       ts_train=None,
                       ts_test=None,
                       filename=None):
    try:

        # Convert train_series into a pandas dataframe and reset index
        df_train = ts_train.pd_dataframe().reset_index()

        # Convert test_series into a pandas dataframe and reset index
        df_test = ts_test.pd_dataframe().reset_index()

        # Convert prediction into a pandas dataframe and reset index
        forecast = predictions.pd_dataframe().reset_index()

        x_feature = 'daily'
        y_feature = 'daily_accident'
        filename = filename
        
        try:
            train_test_predicted_plot(df_train,
                                df_test, 
                                x_feature, 
                                y_feature, forecast,
                                'RNNLSTM',
                                filename=filename)
        except Exception as e:
            print('[ERROR] occurred in train_test_predicted_plot():',e)
    except Exception as e:
        print('Error occurred in plot_visualization function : ', e)
