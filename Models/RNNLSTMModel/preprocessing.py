from darts.dataprocessing.transformers import Scaler

def inverse_transformed(transformer_object,ts_series):
    inversed_tranformed_series = transformer_object.inverse_transform(ts_series)
    return inversed_tranformed_series

def take_prediction(model=None,horizan=None):
    predicted_ts = model.predict(horizan)
    return predicted_ts

def split_train_valid_test(ts_train,split_point=0.8):
    try:
        train_ts, val_ts = ts_train.split_after(split_point)
        return train_ts, val_ts
    except Exception as e:
        print('Error occurred in split_train_valid_test() :',e)



def transformed_ts(ts_train=None,val_subset=None,train_subset=None,train_data_status=False):

    try:

        if train_data_status == True:

            transformer_object = Scaler()
            train_ts_transformed = transformer_object.fit_transform(train_subset)
            val_ts_transformed = transformer_object.transform(val_subset)
            series_ts_transformed = transformer_object.transform(ts_train)

            return transformer_object , train_ts_transformed ,val_ts_transformed ,series_ts_transformed
    except Exception as e:
        print('Error occurred in transformed_ts() : ',e)
