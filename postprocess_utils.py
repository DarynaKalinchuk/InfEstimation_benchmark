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
import math


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



def average_experiments(model_experiments):
        metrics_by_experiment = defaultdict(list)

        for experiments in model_experiments.values():
            for experiment, metrics in experiments:
                metrics_by_experiment[
                    experiment
                ].append(metrics)

        averaged_experiments = []

        for experiment in metrics_by_experiment:
            experiment_metrics = (
                metrics_by_experiment[experiment]
            )
            averaged_metrics = {}

            for key in (
                "accuracy",
                "map",
                "sparsity@5",
                "time_elapsed",
            ):
                finite_values = [
                    metrics[key]
                    for metrics in experiment_metrics
                    if key in metrics
                    and metrics[key] is not None
                    and np.isfinite(metrics[key])
                ]

                averaged_metrics[key] = (
                    float(np.mean(finite_values))
                    if finite_values
                    else np.nan
                )

            averaged_experiments.append(
                (experiment, averaged_metrics)
            )

        return averaged_experiments


def generate_table_metrics(
    source_dir="results/json",
    results_dir="results",
    figsize_scale=0.55,
    max_columns=3,
    method_order = [
        "EKFAC",
        "LiSSA",
        "theta_RelatIF",
        "l_RelatIF",
        "DataInf",
        "TracIn",
        "GradCos",
        "GradDot",
        "random"

    ]
    
):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory not found: {source_dir}")


    files = sorted(
        filename
        for filename in os.listdir(source_dir)
        if filename.endswith(".json")
    )

    if not files:
        raise ValueError(
            f"No JSON files were found in {source_dir!r}."
        )

    grouped = defaultdict(lambda: defaultdict(list))

    for filename in files:
        stem = filename.removesuffix(".json")
        parts = stem.split("_")

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

        grouped[dataset][model].append(
            (experiment, metrics)
        )

    metric_labels = [
        "Accuracy",
        "MAP",
        "Sparsity@5",
        "Runtime",
    ]
    col_names = ["Method", *metric_labels]

    cmap = cm.get_cmap("RdYlGn")
    reverse_cmap = cm.get_cmap("RdYlGn_r")

    os.makedirs(results_dir, exist_ok=True)

    def metrics_to_array(experiments):
        values = np.full(
            (len(experiments), len(metric_labels)),
            np.nan,
        )

        for exp_idx, (_, metrics) in enumerate(experiments):
            values[exp_idx, 0] = metrics.get(
                "accuracy",
                np.nan,
            )
            values[exp_idx, 1] = metrics.get(
                "map",
                np.nan,
            )
            values[exp_idx, 2] = metrics.get(
                "sparsity@5",
                np.nan,
            )
            values[exp_idx, 3] = metrics.get(
                "time_elapsed",
                np.nan,
            )

        return values

    def create_table(ax, title, experiments):
        experiment_labels = [
            experiment
            for experiment, _ in experiments
        ]
        values = metrics_to_array(experiments)

        col_norms = []

        for column_index in range(values.shape[1]):
            if column_index < 3:
                norm = Normalize(
                    vmin=0,
                    vmax=1,
                    clip=True,
                )
            else:
                finite = values[:, column_index][
                    np.isfinite(values[:, column_index])
                ]

                if (
                    len(finite)
                    and finite.min() != finite.max()
                ):
                    norm = Normalize(
                        vmin=finite.min(),
                        vmax=finite.max(),
                        clip=True,
                    )
                else:
                    norm = Normalize(
                        vmin=0,
                        vmax=1,
                        clip=True,
                    )

            col_norms.append(norm)

        cell_colours = [
            [
                (1, 1, 1, 1),
                *[
                    (
                        (1, 1, 1, 1)
                        if np.isnan(value)
                        else (
                            reverse_cmap(
                                col_norms[column_index](
                                    value
                                )
                            )
                            if column_index == 3
                            else cmap(
                                col_norms[column_index](
                                    value
                                )
                            )
                        )
                    )
                    for column_index, value
                    in enumerate(row)
                ],
            ]
            for row in values
        ]

        table_text = [
            [
                experiment_label,
                *[
                    (
                        ""
                        if np.isnan(value)
                        else (
                            f"{value:.2f}s"
                            if column_index == 3
                            else f"{value * 100:.2f}%"
                        )
                    )
                    for column_index, value
                    in enumerate(row)
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

        # Bold method with maximum MAP
        # Also bold the best method among those with strictly higher Accuracy

        accuracy = values[:, 0]
        map_scores = values[:, 1]

        valid = np.isfinite(accuracy) & np.isfinite(map_scores)

        if np.any(valid):
            best_map = np.nanmax(map_scores[valid])

            map_candidates = (
                valid
                & np.isclose(map_scores, best_map)
            )

            best_map_accuracy = np.nanmax(
                accuracy[map_candidates]
            )

            best_map_rows = (
                map_candidates
                & np.isclose(
                    accuracy,
                    best_map_accuracy,
                )
            )

            highlighted_rows = set(
                np.where(best_map_rows)[0]
            )

            selected_idx = next(iter(highlighted_rows))
            selected_accuracy = accuracy[selected_idx]

            higher_accuracy = (
                valid
                & (accuracy > selected_accuracy)
            )

            if np.any(higher_accuracy):
                highest_accuracy = np.nanmax(
                    accuracy[higher_accuracy]
                )

                accuracy_candidates = (
                    higher_accuracy
                    & np.isclose(
                        accuracy,
                        highest_accuracy,
                    )
                )

                best_candidate_map = np.nanmax(
                    map_scores[accuracy_candidates]
                )

                best_accuracy_rows = (
                    accuracy_candidates
                    & np.isclose(
                        map_scores,
                        best_candidate_map,
                    )
                )

                highlighted_rows.update(
                    np.where(best_accuracy_rows)[0]
                )

            for row_idx in highlighted_rows:
                table[
                    (row_idx + 1, 0)
                ].get_text().set_weight("bold")

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)

        ax.set_title(
            title.replace("_", " "),
            fontsize=12,
            pad=15,
        )

    

    figures = {}

    # Individual model tables for each dataset.
    for dataset, model_experiments in grouped.items():
        models = sorted(model_experiments)
        num_models = len(models)

        ncols = 2
        nrows = 2

        max_experiments = max(
            len(experiments)
            for experiments
            in model_experiments.values()
        )

        row_height = max(
            4,
            max_experiments * figsize_scale + 2,
        )

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                8 * ncols,
                row_height * nrows,
            ),
            squeeze=False,
        )

        flat_axes = axes.ravel()

        for ax, model in zip(flat_axes, models):
            create_table(
                ax,
                model,
                sorted(
                    model_experiments[model],
                    key=lambda item: method_order.index(item[0]),
                ),
            )

        for ax in flat_axes[num_models:]:
            ax.axis("off")

        fig.suptitle(
            dataset.replace("_", " "),
            fontsize=16,
            y=0.99,
        )

        fig.tight_layout(
            rect=(0, 0, 1, 0.96)
        )

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

        plt.close(fig)

    # Combined figure with one averaged table per dataset.
    datasets = sorted(grouped)

    combined_data = {
        dataset: average_experiments(
            grouped[dataset]
        )
        for dataset in datasets
    }

    max_combined_experiments = max(
        len(experiments)
        for experiments in combined_data.values()
    )

    combined_row_height = max(
        4,
        max_combined_experiments
        * figsize_scale
        + 2,
    )

    combined_fig, combined_axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(
            10,
            combined_row_height * len(datasets),
        ),
        squeeze=False,
    )

    for row_index, dataset in enumerate(datasets):
        create_table(
            combined_axes[row_index, 0],
            (
                f"{dataset} — "
                "Average across all models"
            ),
            sorted(
                combined_data[dataset],
                key=lambda item: method_order.index(item[0]),
            ),
        )

    combined_fig.suptitle(
        "Average Metrics by Dataset",
        fontsize=16,
        y=0.995,
    )

    combined_fig.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    combined_output_path = os.path.join(
        results_dir,
        "combined_metrics_tables.png",
    )

    combined_fig.savefig(
        combined_output_path,
        bbox_inches="tight",
        dpi=300,
    )

    figures["combined"] = combined_fig

    plt.close(fig)





def generate_combined_plots(
    scores_dir="scores",
    results_dir="results/distr_plots",
    name_begin="Backdoor_1",
    max_columns=3,
):
    scores_dir = Path(scores_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    models = sorted(
        path.name
        for path in scores_dir.iterdir()
        if path.is_dir()
    )

    if not models:
        print(f"No model directories found in {scores_dir}.")
        return None

    num_models = len(models)
    ncols = 2
    nrows = 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )

    flat_axes = axes.ravel()
    legend_handles = {}
    generated_models = 0

    for ax, model_n in zip(flat_axes, models):
        cache_dir = scores_dir / model_n
        files = sorted(
            cache_dir.glob(f"{name_begin}*.csv")
        )
        model_generated = False

        for file in files:
            method = file.stem.replace(
                name_begin,
                "",
                1,
            )


            df = pd.read_csv(file, index_col=0)
            X = df.to_numpy(dtype=float)

            values = X.ravel()
            values = values[np.isfinite(values)]

            if len(values) < 2:
                continue

            std = np.std(values, ddof=1)

            if std > 0:
                values = values / std

            if np.all(values == values[0]):
                continue

            kde = gaussian_kde(values)
            x = np.linspace(
                values.min(),
                values.max(),
                1000,
            )

            line, = ax.plot(
                x,
                kde(x),
                lw=2,
                label=method,
            )

            legend_handles.setdefault(
                method,
                line,
            )
            model_generated = True

        ax.set_xlabel("Scores / Std")
        ax.set_ylabel("Density")
        ax.set_title(
            model_n.replace("_", " ")
        )

        if not model_generated:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        generated_models += int(model_generated)

    for ax in flat_axes[num_models:]:
        ax.axis("off")

    if generated_models == 0:
        print(
            f"No plots generated for "
            f"{name_begin}; skipping."
        )
        plt.close(fig)
        return None

    fig.suptitle(
        f"{name_begin.split('_', 1)[0]} KDE Comparison",
        fontsize=16,
        y=0.99,
    )

    labels = list(legend_handles)
    handles = [
        legend_handles[label]
        for label in labels
    ]

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 5),
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.tight_layout(
        rect=(0, 0.08, 1, 0.96)
    )

    output_path = (
        results_dir
        / f"{name_begin}_all_models_diagnostics.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot to: {output_path}")
