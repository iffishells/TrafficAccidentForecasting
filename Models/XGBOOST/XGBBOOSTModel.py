from darts.models.forecasting.xgboost import  XGBModel
def xgboost_model(series=None,
                  lags=None,
                  output_chunk_length=None,
                  multi_models=None,
                  use_static_covariates=None,
                  likelihood =  None,
                  quantiles = None):

    try:
        
        xgb_model= XGBModel(lags=lags,
                        output_chunk_length=output_chunk_length,
                        # add_encoders={
                        #     # 'cycli0c': {'future': ['month']},
                        #     'datetime_attribute': {'future': ['hour', 'dayofweek']},
                        #     # 'position': {'future': ['relative']},
                        #     # 'custom': {'future': [lambda idx: (idx.year - 2013) / 50]},
                        #     'transformer': Scaler()
                        # }, 
                        likelihood=likelihood,
                        quantiles=quantiles,
                        random_state=199, 
                        multi_models=multi_models,
                        use_static_covariates=use_static_covariates
                           
                           
                           )

        xgb_model.fit(series)
        return xgb_model
    except Exception as e:
        print("[Error] in xgb_model function: " + str(e))    
