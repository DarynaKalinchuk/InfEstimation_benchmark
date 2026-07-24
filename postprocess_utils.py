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
        filename
        for filename in os.listdir(source_dir)
        if filename.endswith(".json")
    )

    # Structure:
    # grouped[dataset][model] = [(experiment, metrics), ...]
    grouped = defaultdict(lambda: defaultdict(list))

    for filename in files:
        stem = filename.removesuffix(".json")
        parts = stem.split("_")

        if len(parts) < 2:
            raise ValueError(
                f"Expected filename starting with model_dataset: {filename}"
            )

        dataset = parts[0]
        model = parts[1]

        if len(parts) >= 5:
            experiment = "_".join(parts[2:-2])
        elif len(parts) >= 3:
            experiment = "_".join(parts[2:])
        else:
            experiment = stem

        filepath = os.path.join(source_dir, filename)

        with open(filepath, encoding="utf-8") as file:
            metrics = json.load(file)

        grouped[dataset][model].append((experiment, metrics))

    metric_labels = ["Accuracy", "MAP", "Sparsity@5", "Runtime"]
    col_names = ["Method"] + metric_labels

    cmap = cm.get_cmap("RdYlGn")
    reverse_cmap = cm.get_cmap("RdYlGn_r")

    figures = {}

    for dataset, model_experiments in grouped.items():
        models = sorted(model_experiments)

        if len(models) > 4:
            raise ValueError(
                f"Dataset {dataset!r} has {len(models)} models; "
                "a maximum of four is supported."
            )

        # 1 model: 1x1
        # 2 models: 1x2
        # 3-4 models: 2x2
        if len(models) == 1:
            nrows, ncols = 1, 1
        elif len(models) == 2:
            nrows, ncols = 1, 2
        else:
            nrows, ncols = 2, 2

        max_experiments = max(
            len(experiments)
            for experiments in model_experiments.values()
        )

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                8 * ncols,
                max(4, max_experiments * figsize_scale + 2) * nrows,
            ),
            squeeze=False,
        )

        flat_axes = axes.ravel()

        for ax, model in zip(flat_axes, models):
            experiments = model_experiments[model]
            experiment_labels = [exp for exp, _ in experiments]

            values = np.full(
                (len(experiments), len(metric_labels)),
                np.nan,
            )

            for exp_idx, (_, metrics) in enumerate(experiments):
                values[exp_idx, 0] = metrics.get("accuracy", np.nan)
                values[exp_idx, 1] = metrics.get("map", np.nan)
                values[exp_idx, 2] = metrics.get("sparsity@5", np.nan)
                values[exp_idx, 3] = metrics.get("time_elapsed", np.nan)

            col_norms = []

            for column_index in range(values.shape[1]):
                finite = values[:, column_index][
                    np.isfinite(values[:, column_index])
                ]

                if len(finite) and finite.min() != finite.max():
                    norm = Normalize(finite.min(), finite.max())
                else:
                    norm = Normalize(0, 1)

                col_norms.append(norm)

            cell_colours = [
                [
                    (1, 1, 1, 1),
                    *[
                        (1, 1, 1, 1)
                        if np.isnan(value)
                        else (
                            reverse_cmap(col_norms[column_index](value))
                            if column_index == 3
                            else cmap(col_norms[column_index](value))
                        )
                        for column_index, value in enumerate(row)
                    ],
                ]
                for row in values
            ]

            table_text = [
                [
                    experiment_label,
                    *[
                        ""
                        if np.isnan(value)
                        else (
                            f"{value:.2f}s"
                            if column_index == 3
                            else f"{value * 100:.2f}%"
                        )
                        for column_index, value in enumerate(row)
                    ],
                ]
                for experiment_label, row in zip(
                    experiment_labels,
                    values,
                )
            ]

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
                model.replace("_", " "),
                fontsize=12,
                pad=15,
            )

        # Hide unused subplot in a three-model 2x2 layout.
        for ax in flat_axes[len(models):]:
            ax.axis("off")

        fig.suptitle(
            dataset.replace("_", " "),
            fontsize=16,
            y=0.99,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.96))

        os.makedirs(results_dir, exist_ok=True)

        output_path = os.path.join(
            results_dir,
            f"{dataset}_metrics_tables.png",
        )

        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
        )

        figures[dataset] = fig

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

    fig, ax_kde = plt.subplots(figsize=(7, 5))

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

    ax_kde.set_xlabel("Value / Std")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(f"{name_begin.split('_',1)[0]} {model_n} KDE Comparison")

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