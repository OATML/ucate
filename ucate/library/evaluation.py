import os
import numpy as np
from ucate.library.utils import plotting
from ucate.library.utils import prediction
from ucate.library import scratch


def get_predictions(
    dl,
    model_0,
    model_1,
    model_prop,
    mc_samples,
    test_set,
):
    x, _ = dl.get_test_data(test_set=test_set)
    if model_1 is not None:
        (mu_0, y_0), (mu_1, y_1) = prediction.mc_sample_tl(
            x=x,
            model_0=model_0.mc_sample,
            model_1=model_1.mc_sample,
            mean=dl.y_mean,
            std=dl.y_std,
            mc_samples=mc_samples,
        )
    else:
        t_0 = np.concatenate(
            [
                np.ones((x.shape[0], 1), dtype="float32"),
                np.zeros((x.shape[0], 1), dtype="float32"),
            ],
            -1,
        )
        t_1 = t_0[:, [1, 0]]
        (mu_0, y_0), (mu_1, y_1) = prediction.mc_sample_cevae(
            x=x,
            t=[t_0, t_1],
            model=model_0.mc_sample,
            mean=dl.y_mean,
            std=dl.y_std,
            mc_samples=mc_samples,
        )
    p_t, _ = prediction.mc_sample_2(x=x, model=model_prop.mc_sample)
    p_t = p_t.mean(0)
    return {"mu_0": mu_0, "mu_1": mu_1, "y_0": y_0, "y_1": y_1, "p_t": p_t}


def evaluate(
    dl,
    model_0,
    model_1,
    model_prop,
    mc_samples,
    regression,
    test_set,
    output_dir,
    reject_pcts,
    quantiles=None,
    exclude_population=False,
):
    predictions = get_predictions(
        dl=dl,
        model_0=model_0,
        model_1=model_1,
        model_prop=model_prop,
        mc_samples=mc_samples,
        test_set=test_set,
    )
    pehe_stats, error_stats, quantiles = evaluate_2(
        dl=dl,
        predictions=predictions,
        regression=regression,
        test_set=test_set,
        output_dir=output_dir,
        reject_pcts=reject_pcts,
        quantiles=quantiles,
        exclude_population=exclude_population,
    )
    return pehe_stats, error_stats, quantiles


def evaluate_2(
    dl,
    predictions,
    regression,
    test_set,
    output_dir,
    reject_pcts,
    quantiles=None,
    exclude_population=False,
):
    tag = "test" if test_set else "train"
    _, cate = dl.get_test_data(test_set=test_set)
    cate_pred, predictive_uncrt, epistemic_unct = prediction.cate_measures(
        mu_0=predictions["mu_0"],
        mu_1=predictions["mu_1"],
        y_0=predictions["y_0"],
        y_1=predictions["y_1"],
        regression=regression,
    )
    recommendation_pred = cate_pred.ravel() > 0.0
    recommendation_true = cate > 0.0
    errors = recommendation_pred != recommendation_true
    pehe_prop = []
    errors_prop = []
    pehe_prop_mag = []
    errors_prop_mag = []
    pehe_unct = []
    errors_unct = []
    pehe_unct_altr = []
    errors_unct_altr = []
    pehe_random = []
    errors_random = []
    num_examples = len(cate)
    p_t = predictions["p_t"]
    p_t_0 = p_t[dl.get_t(test_set)[:, 0] > 0.5]
    p_t_1 = p_t[dl.get_t(test_set)[:, 1] > 0.5]
    digit_counts = {
        "Propensity trimming": {
            "9": np.asarray([0] * len(reject_pcts)),
            "2": np.asarray([0] * len(reject_pcts)),
            "other": np.asarray([0] * len(reject_pcts)),
        },
        "Epistemic Uncertainty": {
            "9": np.asarray([0] * len(reject_pcts)),
            "2": np.asarray([0] * len(reject_pcts)),
            "other": np.asarray([0] * len(reject_pcts)),
        },
    }
    if not test_set:
        kde_0 = scratch.get_density_estimator(x=100.0 * p_t_0, bandwidth=2.0)
        kde_1 = scratch.get_density_estimator(x=100.0 * p_t_1, bandwidth=2.0)
    else:
        kde_0, kde_1 = None, None
    for i, pct in enumerate(reject_pcts):
        if not test_set:
            if quantiles is None:
                quantiles = {}
                quantiles["Propensity quantile"] = [
                    np.quantile(p_t, [pct / 2, 1.0 - (pct / 2)])
                ]
                quantiles["kde"] = {"0": kde_0, "1": kde_1}
                overlap_score = get_overlap_score(
                    kde_0=quantiles["kde"]["0"],
                    kde_1=quantiles["kde"]["1"],
                    p_t=p_t,
                    num_0=len(p_t_0),
                    num_1=len(p_t_1),
                )
                quantiles["Propensity trimming"] = [np.quantile(overlap_score, pct)]
                quantiles["Epistemic Uncertainty"] = [
                    np.quantile(epistemic_unct, 1.0 - pct)
                ]
            else:
                quantiles["Propensity quantile"].append(
                    np.quantile(p_t, [pct / 2, 1.0 - (pct / 2)])
                )
                overlap_score = get_overlap_score(
                    kde_0=quantiles["kde"]["0"],
                    kde_1=quantiles["kde"]["1"],
                    p_t=p_t,
                    num_0=len(p_t_0),
                    num_1=len(p_t_1),
                )
                quantiles["Propensity trimming"].append(np.quantile(overlap_score, pct))
                quantiles["Epistemic Uncertainty"].append(
                    np.quantile(epistemic_unct, 1.0 - pct)
                )
        overlap_score = get_overlap_score(
            kde_0=quantiles["kde"]["0"],
            kde_1=quantiles["kde"]["1"],
            p_t=p_t,
            num_0=len(p_t_0),
            num_1=len(p_t_1),
        )
        ind_prop = np.ravel(
            (p_t >= quantiles["Propensity quantile"][i][0])
            * (p_t <= quantiles["Propensity quantile"][i][1])
        )
        ind_prop_mag = (overlap_score >= quantiles["Propensity trimming"][i]).ravel()
        ind_unct = (epistemic_unct <= quantiles["Epistemic Uncertainty"][i]).ravel()
        ind_random = np.random.choice([False, True], ind_unct.shape, p=[pct, 1.0 - pct])
        pehe_prop.append(
            np.sqrt(np.square(cate[ind_prop] - cate_pred[ind_prop]).mean().ravel())
        )
        errors_prop.append(np.sum(errors[ind_prop]).ravel() / num_examples)
        pehe_prop_mag.append(
            np.sqrt(
                np.square(cate[ind_prop_mag] - cate_pred[ind_prop_mag]).mean().ravel()
            )
        )
        errors_prop_mag.append(np.sum(errors[ind_prop_mag]).ravel() / num_examples)
        pehe_unct.append(
            np.sqrt(np.square(cate[ind_unct] - cate_pred[ind_unct]).mean().ravel())
        )
        errors_unct.append(np.sum(errors[ind_unct]).ravel() / num_examples)
        pehe_random.append(
            np.sqrt(np.square(cate[ind_random] - cate_pred[ind_random]).mean().ravel())
        )
        errors_random.append(np.sum(errors[ind_random]).ravel() / num_examples)
        if not regression:
            digit_indices = dl.get_pops(test_set=test_set)
            for k, v in digit_indices.items():
                digit_counts["Propensity trimming"][k][i] += ind_prop_mag[v].sum()
                digit_counts["Epistemic Uncertainty"][k][i] += ind_unct[v].sum()

    pehe_prop = np.asarray(pehe_prop).ravel()
    errors_prop = np.asarray(errors_prop).ravel()
    pehe_prop_mag = np.asarray(pehe_prop_mag).ravel()
    errors_prop_mag = np.asarray(errors_prop_mag).ravel()
    pehe_unct = np.asarray(pehe_unct).ravel()
    errors_unct = np.asarray(errors_unct).ravel()
    pehe_random = np.asarray(pehe_random).ravel()
    errors_random = np.asarray(errors_random).ravel()
    if regression:
        data = (
            {
                "married mother (95% CI)": [
                    cate[dl.get_subpop(test_set)],
                    cate_pred[dl.get_subpop(test_set)],
                    epistemic_unct[dl.get_subpop(test_set)],
                ],
                "unmarried mother (95% CI)": [
                    cate[np.invert(dl.get_subpop(test_set))],
                    cate_pred[np.invert(dl.get_subpop(test_set))],
                    epistemic_unct[np.invert(dl.get_subpop(test_set))],
                ],
            }
            if exclude_population and test_set
            else {"predictions (95% CI)": [cate_pred, cate, epistemic_unct]}
        )
        plotting.error_bars(
            data=data,
            file_name=os.path.join(output_dir, "cate_scatter_{}.png".format(tag)),
        )
    x = (
        {
            "married mother {}".format(tag): predictive_uncrt[dl.get_subpop(test_set)],
            "unmarried mother {}".format(tag): predictive_uncrt[
                np.invert(dl.get_subpop(test_set))
            ],
        }
        if exclude_population and test_set
        else {tag: predictive_uncrt}
    )
    plotting.histogram(
        x=x,
        bins=64,
        alpha=0.5,
        x_label="$\widehat{Var}[Y_1(\mathbf{x}_i) - Y_0(\mathbf{x}_i)]$",
        y_label="Number of individuals",
        x_limit=(None, None),
        file_name=os.path.join(output_dir, "cate_variance_{}.png".format(tag)),
    )
    x = (
        {
            "married mother {}".format(tag): epistemic_unct[dl.get_subpop(test_set)],
            "unmarried mother {}".format(tag): epistemic_unct[
                np.invert(dl.get_subpop(test_set))
            ],
        }
        if exclude_population and test_set
        else {tag: epistemic_unct}
    )
    plotting.histogram(
        x=x,
        bins=64,
        alpha=0.5,
        x_label="$\widehat{I}_{tot}[\mu_1(\mathbf{x}_i), \mu_0(\mathbf{x}_i)]$",
        y_label="Number of individuals",
        x_limit=(None, None),
        file_name=os.path.join(output_dir, "cate_mi_{}.png".format(tag)),
    )
    x = {
        "$t=0$ {}".format(tag): p_t[dl.get_t(test_set)[:, 0] > 0.5],
        "$t=1$ {}".format(tag): p_t[dl.get_t(test_set)[:, 1] > 0.5],
    }
    plotting.histogram(
        x=x,
        bins=128,
        alpha=0.5,
        x_label="$p(t=1 | \mathbf{x})$",
        y_label="Number of individuals",
        x_limit=(0.0, 1.0),
        file_name=os.path.join(output_dir, "cate_propensity_{}.png".format(tag)),
    )
    pehe_stats = {
        "Propensity quantile": [float(v) for v in pehe_prop],
        "Propensity trimming": [float(v) for v in pehe_prop_mag],
        "Epistemic Uncertainty": [float(v) for v in pehe_unct],
        "Random": [float(v) for v in pehe_random],
    }
    error_stats = {
        "Propensity quantile": [float(v) for v in errors_prop],
        "Propensity trimming": [float(v) for v in errors_prop_mag],
        "Epistemic Uncertainty": [float(v) for v in errors_unct],
        "Random": [float(v) for v in errors_random],
    }
    quantiles["counts"] = digit_counts
    return pehe_stats, error_stats, quantiles


def get_overlap_score(kde_0, kde_1, p_t, num_0, num_1):
    s_0 = num_0 * np.exp(kde_0.score_samples(100.0 * p_t).ravel()) / len(p_t)
    s_1 = num_1 * np.exp(kde_1.score_samples(100.0 * p_t).ravel()) / len(p_t)
    return s_0 * s_1 / (s_0 + s_1 + 1e-7)
