import os
import ray
import json

import numpy as np
import pandas as pd

from scipy import stats

from ucate.library import data
from ucate.library import evaluation
from ucate.library.utils import plotting


REJECT_PCTS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.2,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

POLICIES = [
    "Propensity quantile",
    "Propensity trimming",
    "Epistemic Uncertainty",
    "Random",
]


@ray.remote
def evaluate(output_dir):
    predictions_train = np.load(
        os.path.join(output_dir, "predictions_train.npz"), allow_pickle=True
    )
    predictions_test = np.load(
        os.path.join(output_dir, "predictions_test.npz"), allow_pickle=True
    )
    with open(os.path.join(output_dir, "config.json")) as json_file:
        config = json.load(json_file)
    trial = config["trial"]
    dataset_name = config.get("dataset_name")
    regression = dataset_name in ["acic", "ihdp"]
    exclude_population = config.get("exclude_population")
    # Evaluate on training set
    print(f"TRIAL {trial:04d} ")
    print("Training Evaluation")
    dl = data.DATASETS[dataset_name](
        path=config.get("data_dir"),
        trial=trial,
        exclude_population=exclude_population,
    )
    pehe_train, error_train, quantiles = evaluation.evaluate_2(
        dl=dl,
        predictions=predictions_train,
        regression=regression,
        test_set=False,
        output_dir=output_dir,
        reject_pcts=REJECT_PCTS,
        exclude_population=exclude_population,
    )
    print("\n")
    print("Test Evaluation")
    pehe_test, error_test, _ = evaluation.evaluate_2(
        dl=dl,
        predictions=predictions_test,
        regression=regression,
        test_set=True,
        output_dir=output_dir,
        reject_pcts=REJECT_PCTS,
        quantiles=quantiles,
        exclude_population=exclude_population,
    )

    result = {
        "trial": trial,
        "train": {"pehe": pehe_train, "error": error_train},
        "test": {"pehe": pehe_test, "error": error_test},
    }

    with open(os.path.join(output_dir, "result.json"), "w") as outfile:
        json.dump(result, outfile, indent=4, sort_keys=True)

    return result


def build_summary(results, experiment_dir):
    summary = {
        "train": {
            "error": {
                "Propensity quantile": [],
                "Propensity trimming": [],
                "Epistemic Uncertainty": [],
                "Random": [],
            },
            "pehe": {
                "Propensity quantile": [],
                "Propensity trimming": [],
                "Epistemic Uncertainty": [],
                "Random": [],
            },
        },
        "test": {
            "error": {
                "Propensity quantile": [],
                "Propensity trimming": [],
                "Epistemic Uncertainty": [],
                "Random": [],
            },
            "pehe": {
                "Propensity quantile": [],
                "Propensity trimming": [],
                "Epistemic Uncertainty": [],
                "Random": [],
            },
        },
    }
    for result in results:
        for split in ["train", "test"]:
            for metric in ["error", "pehe"]:
                for policy in POLICIES:
                    summary[split][metric][policy].append(result[split][metric][policy])

    with open(os.path.join(experiment_dir, "summary.json"), "w") as outfile:
        json.dump(summary, outfile, indent=4, sort_keys=True)
    return summary


def summarize(summary, experiment_dir):
    for split in ["train", "test"]:
        for metric in ["error", "pehe"]:
            ys = {}
            df = pd.DataFrame(index=POLICIES, columns=REJECT_PCTS)
            for policy in POLICIES:
                arr = np.asarray(summary[split][metric][policy])
                mean_val = np.nanmean(arr, 0)
                ste_val = stats.sem(arr, 0, nan_policy="omit")
                ys.update({policy: (mean_val, ste_val)})
                row = [f"{m:.03f}+-{s:.03f}" for m, s in zip(mean_val, ste_val)]
                df.loc[policy] = row
            plotting.sweep(
                x=REJECT_PCTS,
                ys=ys,
                y_label="$\sqrt{\epsilon_{PEHE}}$"
                if metric == "pehe"
                else "Number of errors / N",
                file_name=os.path.join(experiment_dir, f"{split}_{metric}_sweep.png"),
            )
            df.to_csv(path_or_buf=os.path.join(experiment_dir, f"{split}_{metric}.csv"))
