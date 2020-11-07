import numpy as np
from sklearn.model_selection import train_test_split


def make_train_tune_datasets(
        x,
        y,
        t,
        validation_pct=0.3,
        random_state=1331
):
    x = x.astype('float32')
    y = y.astype('float32')
    t = t.astype('float32')

    x_train, x_tune, y_train, y_tune, t_train, t_tune = train_test_split(
        x, y, t,
        test_size=validation_pct,
        random_state=random_state
    )
    examples_per_treatment = np.sum(t_train, 0)
    return (x_train, y_train, t_train), (x_tune, y_tune, t_tune), examples_per_treatment
