import os
import json
import shutil
import numpy as np
import tensorflow as tf

from ucate.library import data
from ucate.library import models
from ucate.library import evaluation
from ucate.library.utils import plotting


def train(
    job_dir,
    dataset_name,
    data_dir,
    trial,
    exclude_population,
    verbose,
    mode,
    base_filters,
    dropout_rate,
    beta,
    batch_size,
    epochs,
    learning_rate,
    mc_samples,
):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        except RuntimeError as e:
            print(e)
    print("TRIAL {:04d} ".format(trial))
    experiment_name = f"md-{mode}_bf-{base_filters}_dr-{dropout_rate}_beta-{beta}_bs-{batch_size}_lr-{learning_rate}_ep-{exclude_population}"
    output_dir = os.path.join(
        job_dir,
        dataset_name,
        "tarnet",
        experiment_name,
        f"trial_{trial:03d}",
    )
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    config = {
        "job_dir": job_dir,
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "exclude_population": exclude_population,
        "trial": trial,
        "mode": mode,
        "base_filters": base_filters,
        "dropout_rate": dropout_rate,
        "beta": beta,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "mc_samples": mc_samples,
    }
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    # Instantiate data loaders
    dl = data.DATASETS[dataset_name](
        path=data_dir, trial=trial, exclude_population=exclude_population
    )
    x_train, y_train, t_train, examples_per_treatment = dl.get_training_data()
    if dataset_name in ["acic", "ihdp"]:
        regression = True
        model_name = "mlp"
        loss = tf.keras.losses.MeanSquaredError()
        error = tf.keras.metrics.MeanAbsoluteError()
    else:
        regression = False
        model_name = "cnn"
        loss = tf.keras.losses.BinaryCrossentropy()
        error = tf.keras.metrics.BinaryAccuracy()
    # Instantiate models
    model = models.TARNet(
        do_convolution=model_name == "cnn",
        num_examples=examples_per_treatment,
        dim_hidden=base_filters,
        regression=regression,
        dropout_rate=dropout_rate,
        beta=beta,
        mode=mode,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=[loss, loss],
        metrics=[error],
        loss_weights=[0.0, 0.0],
    )
    model_checkpoint = os.path.join(checkpoint_dir, "model_0")
    model_prop = models.MODELS[model_name](
        num_examples=sum(examples_per_treatment),
        dim_hidden=base_filters,
        dropout_rate=dropout_rate,
        regression=False,
        depth=2,
    )
    model_prop.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=[
            tf.keras.losses.BinaryCrossentropy(),
            tf.keras.losses.BinaryCrossentropy(),
        ],
        metrics=[tf.metrics.BinaryAccuracy()],
        loss_weights=[0.0, 0.0],
    )
    model_prop_checkpoint = os.path.join(checkpoint_dir, "model_prop")
    # Instantiate trainer
    _ = model.fit(
        [x_train, t_train, y_train],
        [y_train, np.zeros_like(y_train)],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.3,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_checkpoint, save_best_only=True, save_weights_only=True
            ),
            tf.keras.callbacks.EarlyStopping(patience=50),
        ],
        verbose=verbose,
    )
    _ = model_prop.fit(
        [x_train, t_train[:, -1]],
        [t_train[:, -1], np.zeros_like(t_train[:, -1])],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.3,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_prop_checkpoint,
                save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(patience=50),
        ],
        verbose=verbose,
    )
    # Restore best models
    model.load_weights(model_checkpoint)
    model_prop.load_weights(model_prop_checkpoint)

    predictions_train = evaluation.get_predictions(
        dl=dl,
        model_0=model,
        model_1=None,
        model_prop=model_prop,
        mc_samples=mc_samples,
        test_set=False,
    )

    predictions_test = evaluation.get_predictions(
        dl=dl,
        model_0=model,
        model_1=None,
        model_prop=model_prop,
        mc_samples=mc_samples,
        test_set=True,
    )

    np.savez(os.path.join(output_dir, "predictions_train.npz"), **predictions_train)
    np.savez(os.path.join(output_dir, "predictions_test.npz"), **predictions_test)

    _, cate = dl.get_test_data(test_set=True)
    plotting.error_bars(
        data={
            "predictions (95% CI)": [
                (predictions_test["mu_1"] - predictions_test["mu_0"]).mean(0).ravel(),
                cate,
                2
                * (predictions_test["mu_1"] - predictions_test["mu_0"]).std(0).ravel(),
            ]
        },
        file_name=os.path.join(output_dir, "cate_scatter_test.png"),
    )
    shutil.rmtree(checkpoint_dir)
