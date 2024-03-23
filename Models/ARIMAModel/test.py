from statsmodels.tsa.stattools import adfuller
from darts.utils.statistics import check_seasonality


def adfuller_test(values):
    '''
    ARIMA models rely on the assumption that the time series being modeled is stationary.
    Therefore that assumption needs to hold if you want to use these models.
    The ARIMA model uses differenced data to make the data stationary, which means there’s a consistency of the data over time.
    This function removes the effect of trends or seasonality, such as market or economic data. We make the data stationary only in case of ARIMA because the ARIMA model looks at the past data to predict future values.
    :param values:
    :return:
    '''
    result=adfuller(values)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("P value is less than 0.05 that means we can reject the null hypothesis(Ho). Therefore we can conclude that data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis that means time series has a unit root which indicates that it is non-stationary ")



def inspect_seasonality(ts):
    try:

        seasonal_period = []
        for m in range(2, 25):
            is_seasonal, period = check_seasonality(ts, m=m, alpha=0.05)
            if is_seasonal:
                print("There is seasonality of order {}.".format(period))
                seasonal_period.append(period)

        print(f'There is seasonality of order : {seasonal_period}')
        return seasonal_period
    except Exception as e:
        print(f'Error : {e}')