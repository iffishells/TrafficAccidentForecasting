from darts import TimeSeries

def preprocess_df_to_ts(df=None, datetime_column_name=None, target_column_name=None, sampling_freq=None):
    # fill_missing_dates=True,fillna_value=True

    try:

        if df is not None and datetime_column_name is not None and target_column_name is not None and sampling_freq is not None:
            # converting dataframe to time series object to make the data to fit the model
            time_series = TimeSeries.from_dataframe(df, datetime_column_name, target_column_name, freq=sampling_freq)
            return time_series
    except Exception as e:
        print('Error :', e)