import os
import sys
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from darts.utils.statistics import plot_acf, plot_pacf


def plot_auto_correlation_plot(ts, m=None, max_lag=None,saving_path=None):
    '''
        The autocorrelation function (ACF) is used to identify the order of ARIMA models.
        The ACF plot shows the correlation between the time series and its lagged version.
        The lag at which the ACF plot crosses the upper confidence interval for the first time is
        considered as the order of the MA component of the ARIMA model. Similarly, if the ACF plot decays slowly,
        it indicates that there is a high degree of autocorrelation in the time series, which means that
         an AR component should be included in the ARIMA model.


    :param ts:
    :param m:
    :param max_lag:
    :return:
    '''
    creating_folder_path =f'{saving_path}/arima_models_plots'
    os.makedirs(creating_folder_path, exist_ok=True)
    plot_acf(ts, m=m, max_lag=max_lag, fig_size=(10, 5), axis=None, default_formatting=True)
    plt.xlabel('lags')
    plt.ylabel('correlation')
    plt.title('Auto Correlation Plot')
    plt.savefig(f'{creating_folder_path}/auto_correlation_plot.png')
    # plt.close()
    plt.show()

def plot_partial_auto_correlation_plot(ts, m=None, max_lag=None,saving_path=None):
    '''
        The partial autocorrection function (PACF) is also used to identify the order of ARIMA models.
        The PACF plot shows the correlation between the time series and its lagged version, but with the influence of the
        intermediate lags removed. The lag at which the PACF plot crosses the upper confidence interval for the first time
        is considered as the order of the AR component of the ARIMA model.


    :param ts:
    :param m:
    :param max_lag:
    :return:
    '''

    creating_folder_path =f'{saving_path}/arima_models_plots'
    os.makedirs(creating_folder_path, exist_ok=True)
    plot_pacf(ts, m=m, max_lag=max_lag, fig_size=(10, 5), axis=None, default_formatting=True)
    plt.xlabel('lags')
    plt.ylabel('correlation')
    plt.title('Partial Auto Correlation Plot')
    # plt.sh ow()
    # plt.savefig(saving_path)
    plt.savefig(f'{creating_folder_path}/partial_auto_correlation_plot.png')
    plt.show()
    # plt.close()
    


import plotly.graph_objects as go


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
                name=f'Sarima Model Prediction',
                mode='lines+markers'
            )
        )

        # Update xaxis properties
        fig.update_xaxes(title_text='Time')

        # Update yaxis properties
        fig.update_yaxes(title_text=y_feature)

        # Update title and height
        title = f'Forecasting using {model_name}\ninput window size :{df_train.shape[0]}\n Horizan : {df_test.shape[0]}'
        fig.update_layout(
            title=title,
            height=500,
            width=1500,
            legend=dict(x=0, y=1, traceorder='normal', orientation='h')
        )

        # Save the plot as an HTML file
        # fig.show()
        parent_path = os.path.join('..','Plots','SARIMAModel', 'Experiments')
        os.makedirs(parent_path,exist_ok=True)
        # print('parent path : ',parent_path)
        # fig.write_html(f'{parent_path}/forecasting_using_{model_name}_combination_{filename}'+'.html')
        fig.write_image(f'{parent_path}/forecasting_using_{model_name}_combination_{filename}' + '.png')
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
        model_name = 'SArima Prediction'
        filename = filename
        train_test_predicted_plot(df_train, df_test, x_feature, y_feature, forecast, 'SARIMA-Prediction',
                                  filename=filename)
    except Exception as e:
        print('Error occurred in plot_visualization function : ', e)
