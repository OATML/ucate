import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="colorblind")

_FONTSIZE = 18
_LEGEND_FONTSIZE = 16
_TICK_FONTSIZE = 16
_MARKER_SIZE = 3.0
_LINEWIDTH = 3.0
_DPI = 300


def error_bars(data, file_name):
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(8, 6))
    min_y = 10000
    min_x = 10000
    max_y = -10000
    max_x = -10000
    max_sigma = -10000
    for k, v in data.items():
        x = v[0]
        y = v[1]
        two_sigma = 2.0 * np.sqrt(v[2])
        _ = plt.errorbar(
            x=x,
            y=y,
            yerr=two_sigma,
            alpha=0.9,
            linestyle="None",
            marker="o",
            elinewidth=1.0,
            capsize=2.0,
            markersize=_MARKER_SIZE,
            label=k,
        )
        min_y = y.min() if y.min() < min_y else min_y
        min_x = x.min() if x.min() < min_x else min_x
        max_y = y.max() if y.max() > max_y else max_y
        max_x = x.max() if x.max() > max_x else max_x
        max_sigma = two_sigma.max() if two_sigma.max() > max_sigma else max_sigma
    _ = plt.xlabel("True CATE", fontsize=_FONTSIZE)
    _ = plt.ylabel("Predicted CATE", fontsize=_FONTSIZE)
    _ = plt.xticks(fontsize=_TICK_FONTSIZE)
    _ = plt.yticks(fontsize=_TICK_FONTSIZE)
    limits = (min(min_y, min_x) - max_sigma, max(max_y, max_x) + max_sigma)
    _ = plt.plot(
        np.arange(limits[0], limits[1] + 1),
        np.arange(limits[0], limits[1] + 1),
        linestyle="--",
        label="ideal prediction line",
    )
    _ = plt.legend(frameon=True, fontsize=_LEGEND_FONTSIZE)
    _ = plt.savefig(file_name, dpi=_DPI)
    plt.close(fig)


def sweep(x, ys, y_label, file_name):
    plt.rcParams["figure.constrained_layout.use"] = True
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    append = "error" in file_name
    x = x + [1.0] if append else x
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(8, 6))
    for k, v in ys.items():
        val, error = v
        val = np.append(val, 0.0) if append else val
        error = np.append(error, 0.0) if append else error
        plot = plt.plot(
            x, val, label=k, linewidth=_LINEWIDTH, linestyle=linestyles.pop(0)
        )
        fill = plt.fill_between(x, val - error, val + error, alpha=0.2)
    _ = plt.xlabel("Proportion of recommendations withheld", fontsize=_FONTSIZE)
    _ = plt.ylabel(y_label, fontsize=_FONTSIZE)
    _ = plt.xticks(fontsize=_TICK_FONTSIZE)
    _ = plt.yticks(fontsize=_TICK_FONTSIZE, rotation=45)
    leg = plt.legend(
        title="Rejection policy",
        loc="upper right",
        frameon=True,
        fontsize=_LEGEND_FONTSIZE,
    )
    leg._legend_box.align = "left"
    plt.setp(leg.get_title(), fontsize=_LEGEND_FONTSIZE)
    _ = plt.savefig(file_name, dpi=_DPI)
    plt.close(fig)


def histogram(
    x,
    bins=50,
    alpha=1.0,
    x_label=None,
    y_label=None,
    x_limit=(0.0, 1.0),
    file_name=None,
):
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.clf()
    plt.cla()
    colors = ["C0", "C1", "C2", "C3"]
    fig = plt.figure(figsize=(8, 6))
    values = [v.ravel() for v in x.values()]
    _ = plt.hist(
        values,
        bins=bins,
        alpha=1.0,
        color=colors[: len(values)],
        label=list(x.keys()),
        linewidth=0.0,
    )

    _ = plt.legend(loc="upper right", frameon=True, fontsize=_LEGEND_FONTSIZE)
    _ = plt.xlabel(x_label, fontsize=_FONTSIZE)
    _ = plt.ylabel(y_label, fontsize=_FONTSIZE)
    _ = plt.xticks(fontsize=_TICK_FONTSIZE)
    _ = plt.yticks(fontsize=_TICK_FONTSIZE, rotation=45)
    _ = plt.xlim(x_limit)
    if file_name is None:
        _ = plt.show()
    else:
        _ = plt.savefig(file_name, dpi=_DPI)
    plt.close(fig)
