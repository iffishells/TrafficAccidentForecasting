from darts.models import RNNModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def RNNLSTMModel(train_ts_transformed,val_ts_transformed):
    try:
        # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
        # a period of 5 epochs (`patience`)
        my_stopper = EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=0.05,
            mode='min',
        )


        my_model = RNNModel(
        model="LSTM",
        hidden_dim=20,
        dropout=0,
        batch_size=16,
        n_epochs=2,
        optimizer_kwargs={"lr": 1e-3},
        model_name="RNNLSTMForecasting",
        log_tensorboard=True,
        random_state=42,
        training_length=20,
        input_chunk_length=14,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs = {"accelerator": "cpu",
                            #  "auto_select_gpus": True,
                            "callbacks": [my_stopper]
                            }
        )
#        {"accelerator": "cpu"} for CPU,

#       {"accelerator": "gpu", "devices": [i]} to use only GPU i (i must be an integer),

#       {"accelerator": "gpu", "devices": -1, "auto_select_gpus": True} to use all available GPUS.
        my_model.fit(
            train_ts_transformed,
            # future_covariates=covariates,
            val_series=val_ts_transformed,
            # val_future_covariates=covariates,
            verbose=True,
        )

        return my_model
    except Exception as e:
        print('Error in RNNLSTMModel() : ',e)