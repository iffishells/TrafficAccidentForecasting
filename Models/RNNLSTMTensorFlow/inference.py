import numpy as np

def forecast(model, X_test, forecast_horizon):
    forecast = []
    current_batch = X_test
    
    for i in range(len(X_test)+1):
        current_pred = model.predict(current_batch.reshape(1, -1, 1))
        # print('Current predition : ',current_pred)
        forecast.append(current_pred)
        current_batch = np.append(current_batch[10:], current_pred)


    forecast_values = forecast[:forecast_horizon]
    forecast_values_flat = np.array(forecast_values).flatten()

    return forecast_values_flat
