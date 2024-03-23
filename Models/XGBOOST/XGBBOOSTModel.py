from darts.models.forecasting.xgboost import  XGBModel
def xgboost_model(series):
    try:
        
        xgb_model= XGBModel(lags=30, 
                        output_chunk_length=24, 
                        # add_encoders={
                        #     # 'cycli0c': {'future': ['month']},
                        #     'datetime_attribute': {'future': ['hour', 'dayofweek']},
                        #     # 'position': {'future': ['relative']},
                        #     # 'custom': {'future': [lambda idx: (idx.year - 2013) / 50]},
                        #     'transformer': Scaler()
                        # }, 
                        # likelihood='poisson', 
                        # quantiles=[0.2, 0.5, 0.75, 0.9], 
                        random_state=199, 
                        multi_models=True, 
                        use_static_covariates=True)

        xgb_model.fit(series)
        return xgb_model
    except Exception as e:
        print("[Error] in xgb_model function: " + str(e))    
