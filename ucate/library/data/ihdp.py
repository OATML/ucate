import os
import numpy as np


class IHDP(object):
    def __init__(
            self,
            path,
            trial,
            center=False,
            exclude_population=False,
    ):
        self.trial = trial
        train_dataset = np.load(
            os.path.join(path, 'ihdp_npci_1-1000.train.npz')
        )
        test_dataset = np.load(
            os.path.join(path, 'ihdp_npci_1-1000.test.npz')
        )
        self.train_data = get_trial(
            dataset=train_dataset,
            trial=trial,
            training=True,
            exclude_population=exclude_population
        )
        self.y_mean = self.train_data['y'].mean(dtype='float32')
        self.y_std = self.train_data['y'].std(dtype='float32')
        self.test_data = get_trial(
            dataset=test_dataset,
            trial=trial,
            training=False,
            exclude_population=exclude_population
        )
        self.dim_x_cont = self.train_data['x_cont'].shape[-1]
        self.dim_x_bin = self.train_data['x_bin'].shape[-1]
        self.dim_x = self.dim_x_cont + self.dim_x_bin

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

    def get_subpop(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        return _data['ind_subpop']

    def get_t(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        return _data['t']

    def preprocess(self, dataset):
        x = np.hstack([dataset['x_cont'], dataset['x_bin']])
        y = (dataset['y'].astype('float32') - self.y_mean) / self.y_std
        t = dataset['t'].astype('float32')
        return x, y, t


def get_trial(
        dataset,
        trial,
        training=True,
        exclude_population=False
):
    bin_feats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    cont_feats = [i for i in range(25) if i not in bin_feats]
    ind_subpop = dataset['x'][:, bin_feats[2], trial].astype('bool')
    x = dataset['x'][:, :, trial]
    if exclude_population:
        x = np.delete(x, bin_feats[2], axis=-1)
        bin_feats.pop(2)
        if training:
            idx_included = np.where(ind_subpop)[0]
        else:
            idx_included = np.arange(dataset['x'].shape[0], dtype='int32')
    else:
        idx_included = np.arange(dataset['x'].shape[0], dtype='int32')
    x_bin = dataset['x'][:, bin_feats, trial][idx_included]
    x_bin[:, 7] -= 1.
    t = dataset['t'][:, trial]
    t_in = np.zeros((len(t), 2), 'float32')
    t_in[:, 0] = 1 - t
    t_in[:, 1] = t
    trial_data = {
        'x_bin': x_bin.astype('float32'),
        'x_cont': dataset['x'][:, cont_feats, trial][idx_included].astype('float32'),
        'y': dataset['yf'][:, trial][idx_included],
        't': t_in[idx_included],
        'ycf': dataset['ycf'][:, trial][idx_included],
        'mu0': dataset['mu0'][:, trial][idx_included],
        'mu1': dataset['mu1'][:, trial][idx_included],
        'ate': dataset['ate'],
        'yadd': dataset['yadd'],
        'ymul': dataset['ymul'],
        'ind_subpop': ind_subpop[idx_included]
    }
    return trial_data
