import os
import shutil
import numpy as np
import tensorflow as tf

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from ucate.library import data
from ucate.library import models
from ucate.library.utils import plotting


def train(
    output_dir,
    dataset_name,
    data_dir,
    trial,
    exclude_population,
    verbose,
):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        except RuntimeError as e:
            print(e)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    dbarts = importr("dbarts")
    # Instantiate data loaders
    dl = data.DATASETS[dataset_name](
        path=data_dir, trial=trial, exclude_population=exclude_population
    )
    x_train, y_train, t_train, examples_per_treatment = dl.get_training_data()
    x_test, cate = dl.get_test_data(test_set=True)
    num_train = len(x_train)
    num_test = len(x_test)
    x_train = np.reshape(x_train, (num_train, -1))
    x_test = np.reshape(x_test, (num_test, -1))
    xt_train = np.hstack([x_train, t_train[:, -1:]])
    xt_test = np.vstack(
        [
            np.hstack([x_train, np.zeros((num_train, 1), "float32")]),
            np.hstack([x_train, np.ones((num_train, 1), "float32")]),
            np.hstack([x_test, np.zeros((num_test, 1), "float32")]),
            np.hstack([x_test, np.ones((num_test, 1), "float32")]),
        ]
    )
    # Instantiate models
    model = dbarts.bart(xt_train, y_train, xt_test, verbose=verbose)
    model_dict = dict(zip(model.names, map(list, list(model))))
    model_prop = models.MODELS["cnn" if dataset_name == "cemnist" else "mlp"](
        num_examples=sum(examples_per_treatment),
        dim_hidden=200,
        dropout_rate=0.5,
        regression=False,
        depth=2,
    )
    model_prop.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.metrics.BinaryAccuracy()],
        loss_weights=[0.0, 0.0],
    )
    model_prop_checkpoint = os.path.join(checkpoint_dir, "model_prop")
    # Instantiate trainer
    _ = model_prop.fit(
        [x_train, t_train[:, -1]],
        [t_train[:, -1], np.zeros_like(t_train[:, -1])],
        batch_size=100,
        epochs=2000,
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
    model_prop.load_weights(model_prop_checkpoint)
    # Predict ys
    y_hat = (
        np.asarray(model_dict["yhat.test"])
        .reshape((2 * num_train + 2 * num_test, -1))
        .transpose()
    )
    y_0_train = y_hat[:, :num_train] * dl.y_std + dl.y_mean
    y_1_train = y_hat[:, num_train : 2 * num_train] * dl.y_std + dl.y_mean
    y_0_test = y_hat[:, 2 * num_train : 2 * num_train + num_test] * dl.y_std + dl.y_mean
    y_1_test = (
        y_hat[:, 2 * num_train + num_test : 2 * num_train + 2 * num_test] * dl.y_std
        + dl.y_mean
    )
    # Predict propensity
    p_t_train, _ = model_prop.predict(
        [x_train, np.zeros((num_train,), "float32")],
        batch_size=200,
        workers=8,
        use_multiprocessing=True,
    )
    p_t_test, _ = model_prop.predict(
        [x_test, np.zeros((num_test,), "float32")],
        batch_size=200,
        workers=8,
        use_multiprocessing=True,
    )
    predictions_train = {
        "mu_0": y_0_train,
        "mu_1": y_1_train,
        "y_0": y_0_train,
        "y_1": y_1_train,
        "p_t": p_t_train,
    }
    predictions_test = {
        "mu_0": y_0_test,
        "mu_1": y_1_test,
        "y_0": y_0_test,
        "y_1": y_1_test,
        "p_t": p_t_test,
    }

    np.savez(os.path.join(output_dir, "predictions_train.npz"), **predictions_train)
    np.savez(os.path.join(output_dir, "predictions_test.npz"), **predictions_test)

    plotting.error_bars(
        data={
            "predictions (95% CI)": [
                (y_1_test - y_0_test).mean(0),
                cate,
                2 * (y_1_test - y_0_test).std(0),
            ]
        },
        file_name=os.path.join(output_dir, "cate_scatter_test.png"),
    )
    plotting.histogram(
        x={
            "$t=0$ test": p_t_test[dl.get_t(True)[:, 0] > 0.5],
            "$t=1$ test": p_t_test[dl.get_t(True)[:, 1] >= 0.5],
        },
        bins=128,
        alpha=0.5,
        x_label="$p(t=1 | \mathbf{x})$",
        y_label="Number of individuals",
        x_limit=(0.0, 1.0),
        file_name=os.path.join(output_dir, "cate_propensity_test.png"),
    )
    shutil.rmtree(checkpoint_dir)
    return -1


def install():
    # Install BART
    robjects.r.options(download_file_method="curl")
    numpy2ri.activate()
    rj = importr("rJava", robject_translations={".env": "rj_env"})
    rj._jinit(parameters="-Xmx16g", force_init=True)
    package_names = ["dbarts"]
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=0)
    utils.chooseCRANmirror(ind=0)
    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(
            StrVector(names_to_install), repos="http://cran.us.r-project.org"
        )
