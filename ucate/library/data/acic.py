import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn import model_selection


class ACIC(object):
    def __init__(
            self,
            path,
            trial,
            center=False,
            exclude_population=False,
    ):
        self.trial = trial
        x_df, targets_df = load_data(
            path=path,
            trial=trial,
            subset='1'
        )
        train_dataset, test_dataset = train_test_split(
            x_df=x_df,
            targets_df=targets_df,
            trial=trial,
            test_size=0.3
        )
        self.train_data = get_trial(
            dataset=train_dataset
        )
        self.x_mean = self.train_data['x_cont'].mean(0, keepdims=True)
        self.x_std = self.train_data['x_cont'].std(0, keepdims=True) + 1e-7
        self.y_mean = self.train_data['y'].mean(dtype='float32')
        self.y_std = self.train_data['y'].std(dtype='float32') + 1e-7
        self.test_data = get_trial(
            dataset=test_dataset
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
        x_cont = (dataset['x_cont'] - self.x_mean) / self.x_std
        x_bin = dataset['x_bin']
        x = np.hstack([x_cont, x_bin])
        y = (dataset['y'].astype('float32') - self.y_mean) / self.y_std
        t = dataset['t'].astype('float32')
        return x, y, t


def load_data(
        path,
        trial,
        subset='1'
):
    x_path = os.path.join(path, 'x.csv')
    targets_dir = os.path.join(path, str(trial + 1))
    targets_paths = os.listdir(targets_dir)
    targets_paths.sort()
    x_df = pd.read_csv(
        x_path
    )
    x_df['x_2'] = [ord(x) - 65 for x in x_df['x_2']]
    x_df['x_21'] = [ord(x) - 65 for x in x_df['x_21']]
    x_df['x_24'] = [ord(x) - 65 for x in x_df['x_24']]
    targets_df = pd.read_csv(
        os.path.join(
            targets_dir,
            targets_paths[0]
        )
    )
    return x_df, targets_df


def train_test_split(
        x_df,
        targets_df,
        trial,
        test_size=0.3,
):
    x_df_train, x_df_test, targets_df_train, targets_df_test = model_selection.train_test_split(
        x_df,
        targets_df,
        test_size=test_size,
        random_state=trial,
        shuffle=True
    )
    train_data = {
        'x': x_df_train,
        'targets': targets_df_train
    }
    test_data = {
        'x': x_df_test,
        'targets': targets_df_test
    }
    return train_data, test_data


def get_trial(
        dataset
):
    cat_feats = {'x_2': 6, 'x_21': 16, 'x_24': 5}
    bin_feats = ['x_17', 'x_22', 'x_38', 'x_51', 'x_54']
    cont_feats = []
    for i in range(1, 59):
        feat_id = 'x_{}'.format(i)
        if (feat_id not in bin_feats) and (feat_id not in cat_feats.keys()):
            cont_feats.append(feat_id)
    x_df = dataset['x']
    x_bin = x_df[bin_feats].to_numpy('float32')
    for k, v in cat_feats.items():
        f = dataset['x'][k].to_numpy()
        f = to_categorical(
            f,
            num_classes=v,
            dtype='float32'
        )
        x_bin = np.hstack([x_bin, f])
    x_cont = x_df[cont_feats].to_numpy('float32')
    targets_df = dataset['targets']
    t = targets_df['z'].to_numpy()
    y0 = targets_df['y0'].to_numpy()
    y1 = targets_df['y1'].to_numpy()
    y = np.zeros_like(t, 'float32')
    y[t > 0.5] = y1[t > 0.5]
    y[t < 0.5] = y0[t < 0.5]
    t_in = np.zeros((len(t), 2), 'float32')
    t_in[:, 0] = 1 - t
    t_in[:, 1] = t
    mu0 = targets_df['mu0'].to_numpy()
    mu1 = targets_df['mu1'].to_numpy()
    trial_data = {
        'x_cont': x_cont,
        'x_bin': x_bin,
        'y': y.astype('float32'),
        't': t_in.astype('float32'),
        'mu0': mu0.astype('float32'),
        'mu1': mu1.astype('float32')
    }
    return trial_data
