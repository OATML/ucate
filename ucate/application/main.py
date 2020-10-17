import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import ray
import json
import click
import tensorflow as tf

from ucate.application import workflows


@click.group(chain=True)
@click.pass_context
def cli(context):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    context.obj = {"n_gpu": len(gpus)}


@cli.command("train")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="local or GCS location for writing checkpoints and exporting models",
)
@click.option("--dataset-name", type=str, default="ihdp", help="dataset name")
@click.option(
    "--data-dir",
    type=str,
    default="./data/",
    help="location to write/read dataset, default=.data/",
)
@click.option(
    "--exclude-population",
    default=False,
    type=bool,
    help="exclude population from training, default=False",
)
@click.option("--num-trials", default=1, type=int, help="number of trials, default=1")
@click.option(
    "--gpu-per-trial",
    default=0.0,
    type=float,
    help="number of gpus for each trial, default=0",
)
@click.option(
    "--cpu-per-trial",
    default=1.0,
    type=float,
    help="number of cpus for each trial, default=1",
)
@click.option("--verbose", default=False, type=bool, help="verbosity default=False")
@click.pass_context
def train(
    context,
    job_dir,
    dataset_name,
    data_dir,
    exclude_population,
    num_trials,
    gpu_per_trial,
    cpu_per_trial,
    verbose,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    gpu_per_trial = 0 if context.obj["n_gpu"] == 0 else gpu_per_trial
    context.obj.update(
        {
            "job_dir": job_dir,
            "dataset_name": dataset_name,
            "data_dir": data_dir,
            "exclude_population": exclude_population,
            "num_trials": num_trials,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "verbose": verbose,
        }
    )


@cli.command("bart")
@click.pass_context
def bart(context):
    from ucate.application.workflows import bart

    config = context.obj
    dataset_name = config.get("dataset_name")
    exclude_population = config.get("exclude_population")
    experiment_name = f"ep-{exclude_population}"

    bart.install()

    results = []
    for trial in range(config.get("num_trials")):
        output_dir = os.path.join(
            config.get("job_dir"),
            dataset_name,
            "bart",
            experiment_name,
            f"trial_{trial:03d}",
        )
        os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir
        config["trial"] = trial
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, "w") as fp:
            json.dump(config, fp, indent=4, sort_keys=True)

        results.append(
            bart.train(
                output_dir=output_dir,
                dataset_name=dataset_name,
                data_dir=config.get("data_dir"),
                trial=trial,
                exclude_population=exclude_population,
                verbose=config.get("verbose"),
            )
        )


@cli.command("tarnet")
@click.pass_context
@click.option(
    "--mode", default="tarnet", type=str, help="mode, one of tarnet, dragon, or mmd"
)
@click.option(
    "--base-filters", default=200, type=int, help="base number of filters, default=200"
)
@click.option(
    "--dropout-rate", default=0.5, type=float, help="dropout rate, default=0.0"
)
@click.option(
    "--beta", default=1.0, type=float, help="dragonnet loss param, default=1.0"
)
@click.option(
    "--epochs", type=int, default=2000, help="number of training epochs, default=750"
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=.0001",
)
@click.option(
    "--mc-samples",
    type=int,
    default=100,
    help="number of mc_samples at inference, default=100",
)
def tarnet(
    context,
    mode,
    base_filters,
    dropout_rate,
    beta,
    epochs,
    batch_size,
    learning_rate,
    mc_samples,
):
    config = context.obj
    dataset_name = config.get("dataset_name")
    exclude_population = config.get("exclude_population")

    @ray.remote(
        num_gpus=config.get("gpu_per_trial"), num_cpus=config.get("cpu_per_trial")
    )
    def trainer(**kwargs):
        func = workflows.train_tarnet(**kwargs)
        return func

    results = []
    for trial in range(config.get("num_trials")):
        results.append(
            trainer.remote(
                job_dir=config["job_dir"],
                dataset_name=dataset_name,
                data_dir=config.get("data_dir"),
                trial=trial,
                exclude_population=exclude_population,
                verbose=config.get("verbose"),
                mode=mode,
                base_filters=base_filters,
                dropout_rate=dropout_rate,
                beta=beta,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mc_samples=mc_samples,
            )
        )
    ray.get(results)


@cli.command("tlearner")
@click.pass_context
@click.option(
    "--base-filters", default=200, type=int, help="base number of filters, default=200"
)
@click.option("--depth", default=5, type=int, help="depth of neural network, default=5")
@click.option(
    "--dropout-rate", default=0.5, type=float, help="dropout rate, default=0.0"
)
@click.option(
    "--epochs", type=int, default=2000, help="number of training epochs, default=750"
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=.0001",
)
@click.option(
    "--mc-samples",
    type=int,
    default=100,
    help="number of mc_samples at inference, default=100",
)
def tlearner(
    context,
    base_filters,
    depth,
    dropout_rate,
    epochs,
    batch_size,
    learning_rate,
    mc_samples,
):
    config = context.obj
    dataset_name = config.get("dataset_name")
    exclude_population = config.get("exclude_population")

    @ray.remote(
        num_gpus=config.get("gpu_per_trial"), num_cpus=config.get("cpu_per_trial")
    )
    def trainer(**kwargs):
        func = workflows.train_tlearner(**kwargs)
        return func

    results = []
    for trial in range(config.get("num_trials")):
        results.append(
            trainer.remote(
                job_dir=config["job_dir"],
                dataset_name=dataset_name,
                data_dir=config.get("data_dir"),
                trial=trial,
                exclude_population=exclude_population,
                verbose=config.get("verbose"),
                base_filters=base_filters,
                depth=depth,
                dropout_rate=dropout_rate,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mc_samples=mc_samples,
            )
        )
    ray.get(results)


@cli.command("cevae")
@click.pass_context
@click.option(
    "--dim-latent", default=32, type=int, help="dimension of latent z, default=32"
)
@click.option(
    "--base-filters", default=200, type=int, help="base number of filters, default=200"
)
@click.option(
    "--dropout-rate", default=0.1, type=float, help="dropout rate, default=0.0"
)
@click.option(
    "--beta", default=0.1, type=float, help="dragonnet loss param, default=1.0"
)
@click.option(
    "--negative-sampling",
    default=True,
    type=bool,
    help="Use negative sampling during training, default=True",
)
@click.option(
    "--epochs", type=int, default=1000, help="number of training epochs, default=750"
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    type=float,
    help="learning rate for gradient descent, default=.0001",
)
@click.option(
    "--mc-samples",
    type=int,
    default=100,
    help="number of mc_samples at inference, default=100",
)
def cevae(
    context,
    dim_latent,
    base_filters,
    dropout_rate,
    beta,
    negative_sampling,
    epochs,
    batch_size,
    learning_rate,
    mc_samples,
):
    config = context.obj
    dataset_name = config.get("dataset_name")
    exclude_population = config.get("exclude_population")

    @ray.remote(
        num_gpus=config.get("gpu_per_trial"), num_cpus=config.get("cpu_per_trial")
    )
    def trainer(**kwargs):
        func = workflows.train_cevae(**kwargs)
        return func

    results = []
    for trial in range(config.get("num_trials")):
        results.append(
            trainer.remote(
                job_dir=config["job_dir"],
                dataset_name=dataset_name,
                data_dir=config.get("data_dir"),
                trial=trial,
                exclude_population=exclude_population,
                verbose=config.get("verbose"),
                dim_latent=dim_latent,
                base_filters=base_filters,
                dropout_rate=dropout_rate,
                beta=beta,
                negative_sampling=negative_sampling,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mc_samples=mc_samples,
            )
        )
    ray.get(results)


@cli.command()
@click.option(
    "--experiment-dir",
    type=str,
    required=True,
    help="Location of saved experiment files",
)
def evaluate(
    experiment_dir,
):
    summary_path = os.path.join(experiment_dir, "summary.json")
    if not os.path.exists(summary_path):
        ray.init(
            dashboard_host="127.0.0.1",
            ignore_reinit_error=True,
        )
        results = []
        _, dirs, _ = list(os.walk(experiment_dir))[0]
        for trial_dir in dirs:
            output_dir = os.path.join(experiment_dir, trial_dir)
            if os.path.exists(
                os.path.join(output_dir, "predictions_train.npz")
            ) and os.path.exists(os.path.join(output_dir, "predictions_test.npz")):
                results.append(workflows.evaluate.remote(output_dir))
        summary = workflows.build_summary(
            results=ray.get(results), experiment_dir=experiment_dir
        )
    else:
        with open(summary_path) as summary_file:
            summary = json.load(summary_file)
    workflows.summarize(summary=summary, experiment_dir=experiment_dir)


if __name__ == "__main__":
    cli()
