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
            metrics_by_experiment[experiment].append(metrics)

    averaged_experiments = []

    for experiment in metrics_by_experiment:
        experiment_metrics = metrics_by_experiment[experiment]
        averaged_metrics = {}

        for key in ("accuracy", "map", "sparsity@5", "time_elapsed"):
            finite_values = [
                metrics[key] for metrics in experiment_metrics
                if key in metrics and metrics[key] is not None and np.isfinite(metrics[key])
            ]

            averaged_metrics[key] = float(np.mean(finite_values)) if finite_values else np.nan

        averaged_experiments.append((experiment, averaged_metrics))

    return averaged_experiments



def generate_table_metrics(
    source_dir="results/json",
    results_dir="results",
    method_order=["EKFAC", "LiSSA", "theta_RelatIF", "l_RelatIF", "DataInf",
                  "TracIn", "GradCos", "GradDot", "random"]
):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory not found: {source_dir}")

    files = sorted(f for f in os.listdir(source_dir) if f.endswith(".json"))
    if not files:
        raise ValueError(f"No JSON files were found in {source_dir!r}.")

    grouped = defaultdict(lambda: defaultdict(list))

    for filename in files:
        stem = filename.removesuffix(".json")
        parts = stem.split("_")
        dataset, model = parts[0], parts[1]

        if len(parts) >= 5:
            experiment = "_".join(parts[2:-2])
        elif len(parts) >= 3:
            experiment = "_".join(parts[2:])
        else:
            experiment = stem

        with open(os.path.join(source_dir, filename), encoding="utf-8") as file:
            metrics = json.load(file)

        grouped[dataset][model].append((experiment, metrics))

    metric_labels = ["Accuracy", "MAP", "Sparsity@5", "Runtime"]
    col_names = ["Method", *metric_labels]
    cmap = cm.get_cmap("RdYlGn")
    reverse_cmap = cm.get_cmap("RdYlGn_r")
    os.makedirs(results_dir, exist_ok=True)

    def metrics_to_array(experiments):
        values = np.full((len(experiments), len(metric_labels)), np.nan)

        for i, (_, metrics) in enumerate(experiments):
            values[i, 0] = metrics.get("accuracy", np.nan)
            values[i, 1] = metrics.get("map", np.nan)
            values[i, 2] = metrics.get("sparsity@5", np.nan)
            values[i, 3] = metrics.get("time_elapsed", np.nan)

        return values

    def create_table(ax, title, experiments):
        experiment_labels = [experiment for experiment, _ in experiments]
        values = metrics_to_array(experiments)
        col_norms = []

        for i in range(values.shape[1]):
            if i < 3:
                norm = Normalize(vmin=0, vmax=1, clip=True)
            else:
                finite = values[:, i][np.isfinite(values[:, i])]
                norm = (
                    Normalize(vmin=finite.min(), vmax=finite.max(), clip=True)
                    if len(finite) and finite.min() != finite.max()
                    else Normalize(vmin=0, vmax=1, clip=True)
                )
            col_norms.append(norm)

        cell_colours = [
            [(1, 1, 1, 1), *[
                (1, 1, 1, 1) if np.isnan(value)
                else reverse_cmap(col_norms[i](value)) if i == 3
                else cmap(col_norms[i](value))
                for i, value in enumerate(row)
            ]]
            for row in values
        ]

        table_text = [
            [label, *[
                "" if np.isnan(value)
                else f"{value:.2f}s" if i == 3
                else f"{value * 100:.2f}%"
                for i, value in enumerate(row)
            ]]
            for label, row in zip(experiment_labels, values)
        ]

        ax.axis("off")
        table = ax.table(
            cellText=table_text, cellColours=cell_colours,
            colLabels=col_names, cellLoc="center", loc="center"
        )

        # Boldening criteria
        accuracy, map_scores = values[:, 0], values[:, 1]
        valid = np.isfinite(accuracy) & np.isfinite(map_scores)

        if np.any(valid):
            # best MAP: bold
            best_map = np.nanmax(map_scores[valid])
            best_map_candidates = valid & np.isclose(map_scores, best_map)
            best_map_accuracy = np.nanmax(accuracy[best_map_candidates])
            best_map_rows = best_map_candidates & np.isclose(accuracy, best_map_accuracy)

            bold_idx = np.where(best_map_rows)[0][0]
            bold_accuracy = accuracy[bold_idx]
            table[(bold_idx + 1, 0)].get_text().set_weight("bold")

            lower_map_rows = valid & (map_scores < best_map)

            if np.any(lower_map_rows):
                second_map = np.nanmax(map_scores[lower_map_rows])
                second_map_candidates = lower_map_rows & np.isclose(map_scores, second_map)

                # Among equal second-MAP methods, choose the one with highest Accuracy
                second_map_accuracy = np.nanmax(accuracy[second_map_candidates])
                second_map_rows = second_map_candidates & np.isclose(
                    accuracy, second_map_accuracy
                )
                italic_idx = np.where(second_map_rows)[0][0]

                if accuracy[italic_idx] > bold_accuracy:
                    table[(italic_idx + 1, 0)].get_text().set_weight("bold")

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
        ax.set_title(title.replace("_", " "), fontsize=12, pad=15)

    # Individual model tables for each dataset.
    for dataset, model_experiments in grouped.items():
        models = sorted(model_experiments)
        num_models = len(models)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6), squeeze=False)
        flat_axes = axes.ravel()

        for ax, model in zip(flat_axes, models):
            create_table(
                ax, model,
                sorted(model_experiments[model], key=lambda item: method_order.index(item[0]))
            )

        for ax in flat_axes[num_models:]:
            ax.axis("off")

        fig.suptitle(dataset.replace("_", " "), fontsize=16, y=1.04)
        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.08, top=0.90,
            wspace=0.20, hspace=0.45
        )

        output_path = os.path.join(results_dir, f"{dataset}_metrics_tables.pdf")
        if os.path.exists(output_path):
            os.remove(output_path)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    datasets = sorted(grouped)
    combined_data = {
        dataset: average_experiments(grouped[dataset])
        for dataset in datasets
    }

    combined_fig, combined_axes = plt.subplots(
        nrows=len(datasets), ncols=1, figsize=(7, 9), squeeze=False
    )

    for i, dataset in enumerate(datasets):
        create_table(
            combined_axes[i, 0],
            f"{dataset}",
            sorted(combined_data[dataset], key=lambda item: method_order.index(item[0]))
        )

    combined_fig.subplots_adjust(
        left=0.08, right=0.92, bottom=0.05, top=0.92, hspace=0.55
    )

    combined_output_path = os.path.join(results_dir, "combined_metrics_tables.pdf")
    if os.path.exists(combined_output_path):
        os.remove(combined_output_path)

    combined_fig.savefig(combined_output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return grouped





def generate_combined_plots(scores_dir="scores", results_dir="results/distr_plots", name_begin="Backdoor_1", max_columns=3, num_models = 4):
    scores_dir, results_dir = Path(scores_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    models = sorted(p.name for p in scores_dir.iterdir() if p.is_dir())
    if not models:
        print(f"No model directories found in {scores_dir}.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    flat_axes, legend_handles, generated_models = axes.ravel(), {}, 0

    for ax, model_n in zip(flat_axes, models):
        model_generated = False

        for file in sorted((scores_dir / model_n).glob(f"{name_begin}*.csv")):
            method = file.stem.replace(name_begin, "", 1)
            values = pd.read_csv(file, index_col=0).to_numpy(dtype=float).ravel()
            values = values[np.isfinite(values)]

            if len(values) < 2:
                continue

            std = np.std(values, ddof=1)
            if std > 0:
                values /= std
            if np.all(values == values[0]):
                continue

            kde = gaussian_kde(values)
            x = np.linspace(values.min(), values.max(), 1000)
            line, = ax.plot(x, kde(x), lw=2, label=method)
            legend_handles.setdefault(method, line)
            model_generated = True

        ax.set(xlabel="Scores / Std", ylabel="Density", title=model_n.replace("_", " "))

        if not model_generated:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)

        generated_models += int(model_generated)

    for ax in flat_axes[num_models:]:
        ax.axis("off")

    if not generated_models:
        print(f"No plots generated for {name_begin}; skipping.")
        plt.close(fig)
        return None

    fig.suptitle(f"{name_begin.split('_', 1)[0]}", fontsize=16, y=0.99)

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles),
            loc="lower center",
            ncol=min(len(legend_handles), 5),
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    output_path = results_dir / f"{name_begin}_all_models_diagnostics.pdf"
    if os.path.exists(output_path):
        os.remove(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def plot_method_differences(grouped, results_dir="results"):
    metric_keys = ["accuracy", "map", "sparsity@5"]
    metric_labels = ["Accuracy", "MAP", "Sparsity@5"]
    comparisons = [
        ("l_RelatIF", "DataInf"),
        ("theta_RelatIF", "DataInf"),
        ("GradCos", "GradDot"),
    ]
    colors = ["#2F6BFF", "#8EC5FF", "#C8A2FF"]

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "normalization_impact.pdf")

    if os.path.exists(output_path):
        os.remove(output_path)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    })

    datasets = sorted(grouped)
    averaged = {
        dataset: dict(average_experiments(grouped[dataset]))
        for dataset in datasets
    }

    dataset_spacing = 0.75
    x = np.arange(len(datasets)) * dataset_spacing
    width = 0.16

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(6.2, 6.2),
        sharex=True, sharey=False,
    )

    for metric_idx, (metric_key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[metric_idx]
        subplot_differences = []

        for comparison_idx, (method_a, method_b) in enumerate(comparisons):
            differences = []

            for dataset in datasets:
                methods = averaged[dataset]
                value_a = methods.get(method_a, {}).get(metric_key, np.nan)
                value_b = methods.get(method_b, {}).get(metric_key, np.nan)
                difference = (
                    (value_a - value_b) / value_b
                    if np.isfinite(value_a) and np.isfinite(value_b) and value_b != 0
                    else np.nan
                )
                differences.append(difference)

            subplot_differences.extend(d for d in differences if np.isfinite(d))
            offset = (comparison_idx - 1) * (width + 0.025)

            bars = ax.bar(
                x + offset, differences, width=width,
                label=f"RelDiff( {method_a}, {method_b} )",
                color=colors[comparison_idx],
                edgecolor="white", linewidth=0.5,
            )

            for bar, value in zip(bars, differences):
                if np.isfinite(value):
                    ax.annotate(
                        f"{100 * value:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 2 if value >= 0 else -2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if value >= 0 else "top",
                        fontsize=6,
                    )

        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(metric_label)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{100 * y:.0f}%")
        )

        if subplot_differences:
            limit = max(abs(v) for v in subplot_differences) * 1.15
            if limit == 0:
                limit = 0.01
            limit = np.ceil(limit * 100) / 100
            ax.set_ylim(-limit, limit)
            ax.set_yticks([-limit, limit])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [dataset.replace("_", " ") for dataset in datasets],
        rotation=20, ha="right",
    )

    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3, frameon=False,
    )

    fig.suptitle("Normalization Impact", fontsize=13, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.subplots_adjust(hspace=0.3)

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path