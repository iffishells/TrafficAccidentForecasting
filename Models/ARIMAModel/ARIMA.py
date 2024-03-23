from darts.models.forecasting.arima import ARIMA

def train_arima_model(ts_train,p, d, q, P, D, Q, S):

    try:

        arima_model =  ARIMA(p= p , #, for Auto regressive parameter
                         d=d  , # for difference to make the data is statioanry
                         q = q ,  # for the moving Average,
                         seasonal_order=(P, D, Q, S)
                         )
        arima_model.fit(ts_train)
        return arima_model
    except Exception as e:
        print('Error in train_ariam_model function : ',e)

