import numpy as np
from sklearn.neighbors import KernelDensity


def get_density_estimator(
        x,
        bandwidth=1.0,
        kernel='gaussian'
):
    idx = np.random.choice(np.arange(len(x)), 2000)
    estimator = KernelDensity(
        kernel=kernel,
        bandwidth=bandwidth,
    )
    return estimator.fit(x[idx])
