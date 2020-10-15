import numpy as np


def mc_sample_tl(x, model_0, model_1, mean=0.0, std=1.0, mc_samples=100):
    y_0, h_0 = mc_sample_2(x, model_0, mc_samples)
    y_0 = y_0 * std + mean
    y_1, h_1 = mc_sample_2(x, model_1, mc_samples)
    y_1 = y_1 * std + mean
    return (y_0, h_0), (y_1, h_1)


def mc_sample_cevae(x, t, model, mean=0.0, std=1.0, mc_samples=100):
    y_0, h_0 = mc_sample_2([x, t[0]], model, mc_samples)
    y_0 = y_0 * std + mean
    y_1, h_1 = mc_sample_2([x, t[1]], model, mc_samples)
    y_1 = y_1 * std + mean
    return (y_0, h_0), (y_1, h_1)


def mc_sample(x, model, mc_samples=100):
    return np.asarray(
        [model(x, training=True) for _ in range(mc_samples)], dtype="float32"
    )
    # return np.asarray([model.predict(x, batch_size=200) for _ in range(mc_samples)], dtype='float32')


def mc_sample_2(x, model, mc_samples=100):
    y, h = [], []
    for _ in range(mc_samples):
        y_pred, h_pred = model(x, batch_size=200)
        y.append(y_pred)
        h.append(h_pred)
    return np.asarray(y, dtype="float32"), np.asarray(h, dtype="float32")


def cate_measures(mu_0, mu_1, y_0, y_1, regression):
    cate_pred = (mu_1 - mu_0).mean(0).ravel()
    predictive_uncrt = np.var(y_1 - y_0, 0).ravel()
    epistemic_unct = np.var(mu_1 - mu_0, 0).ravel()
    return cate_pred, predictive_uncrt, epistemic_unct


def total_mi(p_0, p_1):
    return mi(p_0) + mi(p_1)


def mi(p):
    h = entropy(p.mean(0))
    h_cond = entropy(p).mean(0)
    return h - h_cond


def entropy(p):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log((1 - p))


def differential_entropy(sigma):
    eps = 1e-7
    return 0.5 * np.log(2.0 * np.pi) + np.log(sigma + eps) + 0.5
