import numpy as np
from tensorflow.keras.datasets import mnist


class CEMNIST(object):
    def __init__(
            self,
            path,
            trial,
            center=False,
            exclude_population=False,
    ):
        self.trial = trial
        train_dataset, test_dataset = mnist.load_data()
        self.train_data = get_trial_new(
            dataset=train_dataset,
            trial=trial,
            center=center
        )
        self.y_mean = 0.0
        self.y_std = 1.0
        self.test_data = get_trial_new(
            dataset=test_dataset,
            trial=trial,
            center=center
        )
        self.dim_x = (28, 28, 1)

    def get_training_data(self):
        x, y, t = self.preprocess(self.train_data)
        examples_per_treatment = t.sum(0)
        return x, y, t, examples_per_treatment

    def get_test_data(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        x, _, _ = self.preprocess(_data)
        mu1 = _data['mu1'].astype('float32')
        mu0 = _data['mu0'].astype('float32')
        cate = mu1 - mu0
        return x, cate

    def get_t(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        return _data['t']

    def preprocess(self, dataset):
        x = dataset['x'].astype('float32')
        y = (dataset['y'].astype('float32') - self.y_mean) / self.y_std
        t = dataset['t'].astype('float32')
        return x, y, t

    def get_pops(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        return {
            '9': _data['ind_9'],
            '2': _data['ind_2'],
            'other': _data['ind_other']
        }


def get_trial(
        dataset,
        center=False,
        induce_spurrious_correlation=False
):
    x = np.expand_dims(dataset[0], -1) / 255.
    if center:
        x = 2. * x - 1
    y = dataset[1]
    p = 0.5
    t = np.random.choice([0, 1], size=y.shape, p=[1 - p, p])
    ind = np.in1d(y, [2, 6, 9])
    ind_9 = np.in1d(y, 9)
    ind_6 = np.in1d(y, 6)
    ind_2 = np.in1d(y, 2)
    t[ind_9] = 0
    t[ind_2] = 1
    t_in = np.zeros((len(t), 2), 'float32')
    t_in[:, 0] = 1 - t
    t_in[:, 1] = t
    if induce_spurrious_correlation:
        x[np.in1d(t, 1), :2, :2] = 1.
    y = (ind_9 * (1 - t) + ind_6 * t).astype('float32')
    ycf = (ind_9 * t + ind_6 * (1 - t)).astype('float32')
    mu0 = y * (1 - t) + ycf * t
    mu1 = y * t + ycf * (1 - t)
    x = x.astype('float32')
    trial_data = {
        'x': x[ind],
        'y': y[ind],
        't': t_in[ind],
        'ycf': ycf[ind],
        'mu0': mu0[ind],
        'mu1': mu1[ind],
        'yadd': 0.,
        'ymul': 1.
    }
    return trial_data


def get_trial_new(
        dataset,
        trial,
        center=False
):
    rng = np.random.RandomState(trial)
    x = np.expand_dims(dataset[0], -1) / 255.
    if center:
        x = 2. * x - 1
    y = dataset[1]
    p_t = 0.5
    p_t_9 = 1 / 9
    t = rng.choice(
        [0, 1],
        size=y.shape, p=[1 - p_t, p_t],

    )
    ind_9 = np.in1d(y, 9)
    ind_2 = np.in1d(y, 2)
    ind_even = np.in1d(y, [0, 4, 6, 8])
    ind_odd = np.in1d(y, [1, 3, 5, 7])
    num_examples = 2 * ind_9.sum()
    t[ind_9] = rng.choice([0, 1], size=ind_9.sum(), p=[1 - p_t_9, p_t_9])
    t[ind_2] = 1
    t_in = np.zeros((len(t), 2), 'float32')
    t_in[:, 0] = 1 - t
    t_in[:, 1] = t
    p_x = (0.5 / 9) * np.ones_like(y)
    p_x[ind_9] = 0.5
    p_x = p_x / p_x.sum()
    y = np.zeros_like(y, 'float32')
    y[ind_9] = 1 - t[ind_9]
    y[ind_2] = t[ind_2]
    y[ind_even] = t[ind_even]
    y[ind_odd] = 1 - t[ind_odd]
    ycf = np.zeros_like(y, 'float32')
    ycf[ind_9] = t[ind_9]
    ycf[ind_2] = 1 - t[ind_2]
    ycf[ind_even] = 1 - t[ind_even]
    ycf[ind_odd] = t[ind_odd]
    mu0 = y * (1 - t) + ycf * t
    mu1 = y * t + ycf * (1 - t)
    x = x.astype('float32')
    ind = rng.choice(np.arange(len(y)), size=num_examples, p=p_x)
    trial_data = {
        'x': x[ind],
        'y': y[ind],
        't': t_in[ind],
        'ycf': ycf[ind],
        'mu0': mu0[ind],
        'mu1': mu1[ind],
        'yadd': 0.,
        'ymul': 1.,
        'ind_2': ind_2[ind],
        'ind_9': ind_9[ind],
        'ind_other': (ind_even + ind_odd)[ind]
    }
    return trial_data
