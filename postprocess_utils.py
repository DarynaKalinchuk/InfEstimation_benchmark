"""
Evaluation metrics and post-processing of results.
"""
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np
import os
import json
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
from matplotlib.ticker import PercentFormatter
from datasets import load_from_disk
import pandas as pd
from scipy.stats import gaussian_kde


def run_benchmark_measures(
    metrics_path,
    method="random",
    scores_path="scores/..",
    runtime_path="runtime_stats/..",
    dataset_path="datasets/Backdoor",
    field="variation",
):
    if method == "random":
        return None

    dataset = load_from_disk(dataset_path)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    influence = pd.read_csv(scores_path)

    time_elapsed = None
    if runtime_path is not None and os.path.exists(runtime_path):
        with open(runtime_path, "r") as f:
            time_elapsed = float(f.read().strip())

    expected_shape = (
        len(test_dataset),
        len(train_dataset),
    )

    if influence.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, "
            f"got {influence.shape}"
        )

    train_labels = list(train_dataset[field])
    test_labels = list(test_dataset[field])

    train_class_counts = Counter(train_labels)
    M = len(test_dataset)

    overall = {
        "acc": 0,
        "map": 0,
        "sparsity5": 0,
    }

    for i in range(M):
        test_label = test_labels[i]
        scores = influence.iloc[i].to_numpy()

        # Sparsity@5 using positive influence scores
        positive_scores = np.maximum(scores, 0)
        total_sum = positive_scores.sum()

        if total_sum > 0:
            threshold = np.percentile(positive_scores, 95)
            top_sum = positive_scores[positive_scores >= threshold].sum()
            overall["sparsity5"] += top_sum / total_sum

        k = train_class_counts.get(test_label, 0)

        # Full ranking from highest to lowest
        ranking = np.argsort(scores)[::-1]

        # Accuracy
        top1 = int(ranking[0])

        if train_labels[top1] == test_label:
            overall["acc"] += 1

        # Average Precision
        if k > 0:
            hits = 0
            precision_sum = 0.0

            for rank, idx in enumerate(ranking, start=1):
                if train_labels[int(idx)] == test_label:
                    hits += 1
                    precision_sum += hits / rank

            overall["map"] += precision_sum / k

    metrics = {
        "accuracy": overall["acc"] / M,
        "map": overall["map"] / M,
        "sparsity@5": overall["sparsity5"] / M,
    }

    if time_elapsed is not None:
        metrics["time_elapsed"] = time_elapsed

    print("Accuracy:", metrics["accuracy"])
    print("MAP:", metrics["map"])
    print("Sparsity@5:", metrics["sparsity@5"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics



def generate_table_metrics(
    source_dir="results/json",
    results_dir="results",
    figsize_scale=0.55,
    show=True,
):

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory not found: {source_dir}")

    files = sorted(
        f for f in os.listdir(source_dir)
        if f.endswith(".json")
    )

    grouped = defaultdict(list)

    for file in files:
        stem = file.removesuffix(".json")
        parts = stem.split("_")

        group = "_".join(parts[:2]) if len(parts) >= 2 else stem

        if len(parts) >= 5:
            exp = "_".join(parts[2:-2])
        elif len(parts) >= 3:
            exp = "_".join(parts[2:])
        else:
            exp = stem

        with open(os.path.join(source_dir, file)) as f:
            grouped[group].append((exp, json.load(f)))

    metric_labels = ["Accuracy", "MAP", "Sparsity@5", "Runtime"]
    figures = {}

    for group_key, experiments in grouped.items():

        experiment_labels = [exp for exp, _ in experiments]
        col_names = ["Method"] + metric_labels

        values = np.full(
            (len(experiment_labels), len(metric_labels)),
            np.nan,
        )

        for exp_idx, (exp, metrics) in enumerate(experiments):

            values[exp_idx, 0] = metrics.get("accuracy", np.nan)
            values[exp_idx, 1] = metrics.get("map", np.nan)
            values[exp_idx, 2] = metrics.get("sparsity@5", np.nan)
            values[exp_idx, 3] = metrics.get("time_elapsed", np.nan)

        cmap = cm.get_cmap("RdYlGn")
        reverse_cmap = cm.get_cmap("RdYlGn_r")

        col_norms = []

        for c in range(values.shape[1]):
            finite = values[:, c][np.isfinite(values[:, c])]

            if len(finite) and finite.min() != finite.max():
                col_norms.append(
                    Normalize(finite.min(), finite.max())
                )
            else:
                col_norms.append(Normalize(0, 1))

        cell_colours = [
            [
                (1, 1, 1, 1),
                *[
                    (1, 1, 1, 1)
                    if np.isnan(v)
                    else (
                        reverse_cmap(col_norms[c](v))
                        if c == 3
                        else cmap(col_norms[c](v))
                    )
                    for c, v in enumerate(row)
                ],
            ]
            for row in values
        ]

        table_text = [
            [
                exp_label,
                *[
                    ""
                    if np.isnan(v)
                    else f"{v:.2f}s"
                    if c == 3
                    else f"{v * 100:.2f}%"
                    for c, v in enumerate(row)
                ],
            ]
            for exp_label, row in zip(experiment_labels, values)
        ]

        fig, ax = plt.subplots(
            figsize=(
                max(10, len(col_names) * 2),
                max(
                    4,
                    len(experiment_labels) * figsize_scale + 2,
                ),
            )
        )

        ax.axis("off")

        table = ax.table(
            cellText=table_text,
            cellColours=cell_colours,
            colLabels=col_names,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)

        ax.set_title(
            group_key.replace("_", " "),
            fontsize=14,
            pad=20,
        )

        plt.tight_layout()

        os.makedirs(results_dir, exist_ok=True)

        output_path = os.path.join(
            results_dir,
            f"{group_key}_metrics_table.png",
        )

        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
        )

        figures[group_key] = fig

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures



def generate_combined_plots(
    model_n="Olmo",
    results_dir="results/distr_plots",
    name_begin="Backdoor_1",
):

    cache_dir = Path("scores/" + model_n)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cache_dir.glob(name_begin + "*.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    ax_kde, ax_cov = axs

    percentiles = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])

    for file in files:
        method = file.stem.replace(name_begin, "")

        # ignore
        if method == "BM25":
            continue

        df = pd.read_csv(file, index_col=0)
        X = df.to_numpy(dtype=float)

        values = X.ravel()
        values = values[np.isfinite(values)]

        if len(values) >= 2:
            std = np.std(values, ddof=1)
            if std > 0:
                values = values / std

            kde = gaussian_kde(values)
            x = np.linspace(values.min(), values.max(), 1000)
            ax_kde.plot(x, kde(x), lw=2, label=method)


            avg_coverage = []

            for p in percentiles:
                row_coverages = []

                for row in X:
                    row_sorted = np.sort(row)[::-1]

                    total = row[row > 0].sum()
                    if total == 0:
                        continue

                    k = max(1, int(np.ceil(p * len(row_sorted))))
                    coverage = row_sorted[:k].clip(min=0).sum() / total
                    row_coverages.append(coverage)

                avg_coverage.append(
                    np.mean(row_coverages) if row_coverages else np.nan
                )

            ax_cov.plot(percentiles, avg_coverage, marker="o", lw=2, label=method)

    ax_kde.set_xlabel("Value / Std")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(f"{name_begin.split('_',1)[0]} {model_n} KDE Comparison")

    ax_cov.set_xlabel("Top percentile")
    ax_cov.set_ylabel("Average coverage of positive values")
    ax_cov.set_title(f"{name_begin.split('_',1)[0]} {model_n} Coverage Comparison")
    ax_cov.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_cov.grid(True)

    handles, labels = ax_kde.get_legend_handles_labels()

    if not handles:
        print(f"No plots generated for {name_begin} ({model_n}); skipping.")
        plt.close(fig)
        return None

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = results_dir / f"{name_begin}_{model_n}_diagnostics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {output_path}")

    return output_path