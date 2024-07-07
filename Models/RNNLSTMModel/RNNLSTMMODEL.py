from darts.models import RNNModel
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def RNNLSTMModel(train_ts_transformed = None,
                val_ts_transformed= None,
                    model=None,
                    hidden_dim=None,
                    n_rnn_layers=None,
                    dropout=None,
                    batch_size=None,
                    n_epochs=None,
                    learning_rate=None,
                    model_name=None,
                    log_tensorboard=True,
                    random_state=42,
                    training_length=None,
                    input_chunk_length=None,
                    force_reset=True,
                    save_checkpoints=True
                
                
                ):
    try:
        # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
        # a period of 5 epochs (`patience`)
        # my_stopper = EarlyStopping(
        #     monitor="val_loss",
        #     patience=30,
        #     min_delta=0.0001,
        #     mode='min',
        # )


        my_model = RNNModel(
        model=model,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": learning_rate},
        model_name=model_name,
        log_tensorboard=log_tensorboard,
        random_state=42,
        training_length=training_length,
        input_chunk_length=input_chunk_length,
        force_reset=force_reset,
        save_checkpoints=True,
        pl_trainer_kwargs = {"accelerator": "cpu"
                            #  "auto_select_gpus": True,
                            # "callbacks": [my_stopper]
                            }
        # {"accelerator": "gpu",
        #  "devices": -1,
        #  "auto_select_gpus": True}  # to use all available GPUS.

        )
#        {"accelerator": "cpu"} for CPU,

      # {"accelerator": "gpu", "devices": [i]},# to use only GPU i (i must be an integer),

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
