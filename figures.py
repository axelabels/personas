"""Regenerate only figures referenced by paper/main.tex; never call an LLM."""
import argparse
import ast
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from analysis import (
    DATA,
    MODELS,
    COUNTRIES,
    TARGETS,
    METHODS,
    COLORS,
    clean_model_name,
    load_responses,
    human_distributions,
    aggregate,
    error_table,
    calibration,
    normalized_variance,
    variance_aggregate,
    values,
    scale,
)

FIGURES = [
    "heatmap_mae",
    "peak_distributions",
    "guidance_comparison",
    "error_comparison",
    "temperature_all_curves",
    "q66_histograms_ghana",
]


def save_figure(figure, output_dir, name):
    """Save a PDF with consistent bounds and release the plotting resources."""
    figure.savefig(output_dir / (name + ".pdf"), bbox_inches="tight", dpi=150)
    plt.close(figure)


def plot_error_heatmap(errors, output_dir):
    """Compare normalized errors by method, model, and country."""
    sns.set_context("talk", font_scale=0.8)
    table = (
        errors.groupby(["model", "country", "method"])
        .error.mean()
        .rename("value")
        .reset_index()
    )
    country_order = (
        table[table.method == "Generic prompt"]
        .groupby("country")
        .value.mean()
        .sort_values()
        .index.tolist()
    )
    country_averages = (
        table.groupby(["model", "method"])
        .value.mean()
        .reset_index()
        .assign(country="All countries")
    )
    table = pd.concat([table, country_averages], ignore_index=True)
    color_min, color_max = np.percentile(table.value, [5, 95])
    figure, axes = plt.subplots(
        2, 2, figsize=(13.5, 8.4), gridspec_kw={"wspace": 0.05, "hspace": 0.1}
    )
    # Compare the two lowest-error methods using paired question-level errors.
    best_methods = {}
    significance = {}
    for (model, country_key), group in errors.groupby(["model", "country"]):
        method_errors = group.pivot(
            index="question", columns="method", values="error"
        ).dropna()
        ranked_methods = method_errors.mean().sort_values()
        best_methods[model, country_key] = ranked_methods.index[0]
        error_difference = (
            method_errors[ranked_methods.index[0]]
            - method_errors[ranked_methods.index[1]]
        ).to_numpy()
        p_value = (
            wilcoxon(error_difference, alternative="less").pvalue
            if len(error_difference) >= 4 and np.any(error_difference)
            else 1.0
        )
        significance[model, country_key] = (
            2 if p_value < 0.05 else 1 if p_value < 0.1 else 0
        )

    def majority(levels):
        """Summarize significance using the study's majority convention."""
        levels = np.asarray(levels)
        if np.mean(levels == 2) > 0.5:
            return 2
        if np.mean(levels >= 1) > 0.5:
            return 1
        return 0

    for model in MODELS:
        best_methods[model, "All countries"] = (
            table[(table.model == model) & (table.country == "All countries")]
            .set_index("method")
            .value.idxmin()
        )
        significance[model, "All countries"] = majority(
            [significance.get((model, country_key), 0) for country_key in country_order]
        )
    for panel_index, (method, axis) in enumerate(zip(METHODS, axes.flat)):
        matrix = (
            table[table.method == method]
            .pivot(index="model", columns="country", values="value")
            .reindex(index=MODELS, columns=["All countries"] + country_order)
        )
        matrix = pd.concat([pd.DataFrame([matrix.mean()], index=["Avg"]), matrix])
        heatmap_image = axis.imshow(
            matrix, cmap="RdYlGn_r", vmin=color_min, vmax=color_max, aspect="auto"
        )
        for row_index, model in enumerate(matrix.index):
            for column_index, country in enumerate(matrix.columns):
                cell_value = matrix.iloc[row_index, column_index]
                if not np.isfinite(cell_value):
                    continue
                winner = (
                    table[table.country == country]
                    .groupby("method")
                    .value.mean()
                    .idxmin()
                    if model == "Avg"
                    else best_methods.get((model, country))
                )
                level = (
                    majority(
                        [
                            significance.get((other_model, country), 0)
                            for other_model in MODELS
                        ]
                    )
                    if model == "Avg"
                    else significance.get((model, country), 0)
                )
                color = (
                    "white"
                    if (cell_value - color_min) / (color_max - color_min) > 0.75
                    or (cell_value - color_min) / (color_max - color_min) < 0.2
                    else "black"
                )
                axis.text(
                    column_index,
                    row_index,
                    f"{cell_value:.3f}".lstrip("0"),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    fontweight="bold" if winner == method else "normal",
                )
                if winner == method and level:
                    axis.plot(
                        [column_index - 0.32, column_index + 0.32],
                        [row_index + 0.22, row_index + 0.22],
                        color=color,
                        lw=1,
                        linestyle="-" if level == 2 else ":",
                    )
        axis.axhline(0.5, color="white", lw=2)
        axis.axvline(0.5, color="lightgrey", lw=2)
        axis.set_xticks(
            range(len(matrix.columns)),
            [country_key.split(",")[0] for country_key in matrix.columns]
            if panel_index >= 2
            else [""] * len(matrix.columns),
            rotation=45,
            ha="right",
        )
        axis.set_yticks(
            range(len(matrix.index)),
            ["Avg"] + [clean_model_name(model) for model in MODELS]
            if panel_index % 2 == 0
            else [""] * len(matrix.index),
        )
        axis.set_title(method, color=COLORS[method])
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.colorbar(
        heatmap_image,
        ax=list(axes.flat),
        fraction=0.02,
        pad=0.02,
        shrink=0.6,
        label="MAE",
    )
    save_figure(figure, output_dir, "heatmap_mae")


def plot_error_comparison(errors, output_dir):
    """Compare value and country prompting errors for each target question."""
    sns.set_context("poster", font_scale=1.1)
    method_errors = (
        errors.groupby(["model", "question", "method"]).error.mean().unstack("method")
    )
    figure, axis = plt.subplots(figsize=(7, 7))
    markers = ["s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d"]
    for panel_index, model in enumerate(MODELS):
        model_errors = method_errors.loc[model]
        axis.scatter(
            model_errors["Value prompt"],
            model_errors["Country prompt"],
            alpha=0.5,
            marker=markers[panel_index],
            color=f"C{panel_index % 10}",
            label=clean_model_name(model),
        )
    means = method_errors.groupby("question").mean()
    axis.scatter(
        means["Value prompt"],
        means["Country prompt"],
        s=100,
        color="white",
        edgecolor="black",
        label="Mean",
        zorder=5,
    )
    axis_limit = (
        float(method_errors[["Value prompt", "Country prompt"]].max().max()) + 0.01
    )
    axis.plot([0, axis_limit], [0, axis_limit], "k--", alpha=0.5)
    axis.set(
        xlabel="Value prompt MAE",
        ylabel="Country prompt MAE",
        xlim=(0, axis_limit),
        ylim=(0, axis_limit),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.3)
    axis.legend(bbox_to_anchor=(1, 1.15), loc="upper left", frameon=False)
    sns.despine(ax=axis)
    save_figure(figure, output_dir, "error_comparison")


def plot_temperature_curves(curves, output_dir):
    """Plot error metrics under standard and mean-preserving scaling."""
    sns.set_context("poster")
    mean_errors = (
        curves.groupby(["model", "country", "method", "temperature", "temp_method"])[
            ["mae", "wasserstein"]
        ]
        .mean()
        .reset_index()
    )
    figure, axes = plt.subplots(1, 4, figsize=(24, 4), layout="constrained")
    colors = {
        **COLORS,
        "Sociodemographic prompt": "#e07b39",
        "Generic prompt": "#2ca02c",
    }
    for panel_index, axis in enumerate(axes):
        mode = ["Classic", "Mean-preserving"][panel_index // 2]
        metric = ["mae", "wasserstein"][panel_index % 2]
        sns.lineplot(
            data=mean_errors[mean_errors.temp_method == mode],
            x="temperature",
            y=metric,
            hue="method",
            style="method",
            palette=colors,
            seed=0,
            ax=axis,
        )
        axis.axvline(1, color="gray", ls="--")
        axis.set_xscale("log")
        axis.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        axis.set(
            xlabel="Temperature",
            ylabel="MAE" if metric == "mae" else "Wass. distance",
            title="ABCD"[panel_index] + ".",
        )
        handles, labels = axis.get_legend_handles_labels()
        axis.get_legend().remove()
        sns.despine(ax=axis)
    axes[-1].legend(
        handles, labels, bbox_to_anchor=(1, 1), loc="upper left", frameon=False
    )
    figure.text(0.23, 1.05, "Standard Temperature Scaling", ha="center")
    figure.text(0.67, 1.05, "Mean-Preserving Scaling", ha="center")
    save_figure(figure, output_dir, "temperature_all_curves")


def plot_variance(aggregated, human, calibrated, output_dir):
    """Compare human, country-prompt, and value-prompt response variance."""
    sns.set_context("poster")
    names = ["Country prompt", "Value prompt", "Calibrated Value prompt"]
    figure, axes = plt.subplots(1, 3, figsize=(21.6, 5.33), sharey=True)
    human_variances = [
        normalized_variance(human[country, question])
        for country in COUNTRIES
        for question in TARGETS
    ]
    for panel_index, (name, axis) in enumerate(zip(names, np.ravel(axes))):
        source = "Value prompt" if name.startswith("Calibrated") else name
        method_responses = aggregated[aggregated.method == source]
        groups = []
        for model in MODELS:
            model_variances = []
            for row in method_responses[method_responses.model == model].itertuples(
                index=False
            ):
                probabilities = (
                    calibrated[model, source, row.country, row.question]
                    if name.startswith("Calibrated")
                    else row.probabilities
                )
                model_variances.append(normalized_variance(probabilities))
            groups.append(model_variances)
        boxplot = axis.boxplot(
            [human_variances] + groups,
            positions=np.arange(-1, len(MODELS)),
            notch=True,
            showfliers=False,
            patch_artist=True,
            widths=0.5,
        )
        for box_index, box in enumerate(boxplot["boxes"]):
            box.set(
                facecolor=(0, 0, 0, 0.15)
                if box_index == 0
                else matplotlib.colors.to_rgba(f"C{box_index % 10}", 0.5),
                edgecolor="black" if box_index == 0 else f"C{box_index % 10}",
                linewidth=2,
            )
        axis.set_xticks(
            np.arange(-1, len(MODELS)),
            ["Human"] + [clean_model_name(model) for model in MODELS],
            rotation=45,
            ha="right",
        )
        axis.set(
            ylim=(0, 0.7),
            title=f"{chr(65 + panel_index)}. {name.title()}",
            ylabel="Normalized\nVariance" if panel_index % 3 == 0 else "",
        )
        sns.despine(ax=axis, bottom=True)
    figure.subplots_adjust(wspace=0.05)
    save_figure(figure, output_dir, "peak_distributions")


def plot_ghana_distributions(aggregated, human, calibrated, output_dir):
    """Show the Ghana Q66 example before and after calibration."""
    key = ("gpt-5.4-nano", "Value prompt", "Ghana", "Q66")
    selected_response = aggregated[
        (aggregated.model == key[0])
        & (aggregated.method == key[1])
        & (aggregated.country == key[2])
        & (aggregated.question == key[3])
    ]
    if len(selected_response) != 1:
        raise ValueError("Expected one aggregated Ghana Q66 response for gpt-5.4-nano")
    sns.set_context("talk")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, title, probabilities in zip(
        axes,
        ["Human", "Value prompt", "Calibrated Value prompt"],
        [
            human["Ghana", "Q66"],
            selected_response.iloc[0].probabilities,
            calibrated[key],
        ],
    ):
        axis.bar(["1", "2", "3", "4"], probabilities, color="#2a78d6")
        axis.set(title=title, ylim=(0, 1), ylabel="Probability")
        sns.despine(ax=axis)
    figure.tight_layout()
    save_figure(figure, output_dir, "q66_histograms_ghana")


def plot_guidance_comparison(output_dir, human):
    """Compare the recorded guidance variants after battery-level averaging."""
    guidance_responses = pd.read_parquet(DATA / "guidance_responses.parquet")
    rows = []
    aliases = {country.split(",")[0]: country for country in COUNTRIES}
    for row in guidance_responses.itertuples(index=False):
        human_probabilities = human[aliases[row.country], row.qkey]
        if row.qkey in ["Q154", "Q155"]:
            model_probabilities = {
                float(answer): float(probability)
                for answer, probability in ast.literal_eval(row.probs).items()
            }
            reference_probabilities = dict(zip(values(row.qkey), human_probabilities))
            response_values = set(model_probabilities) | set(reference_probabilities)
            error = (
                np.mean(
                    [
                        abs(
                            reference_probabilities.get(answer, 0)
                            - model_probabilities.get(answer, 0)
                        )
                        for answer in response_values
                    ]
                )
                / 2
            )
        else:
            error = abs(human_probabilities @ values(row.qkey) - row.ev) / scale(
                row.qkey
            )
        battery = (
            "autonomy"
            if row.qkey in ["Q8", "Q11", "Q15", "Q17"]
            else "postmaterialist"
            if row.qkey in ["Q154", "Q155"]
            else row.qkey
        )
        rows.append(
            dict(
                model=row.model,
                country=row.country,
                battery=battery,
                guidance=row.guidance,
                error=error,
            )
        )
    guidance_responses = (
        pd.DataFrame(rows)
        .groupby(["model", "country", "battery", "guidance"])
        .error.mean()
        .reset_index()
    )
    order = guidance_responses.groupby("guidance").error.median().sort_values().index
    sns.set_context("talk")
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=guidance_responses,
        y="guidance",
        x="error",
        order=order,
        notch=True,
        showfliers=False,
        ax=axis,
    )
    axis.set(xlabel="MAE", ylabel="Guidance")
    axis.grid()
    figure.tight_layout()
    save_figure(figure, output_dir, "guidance_comparison")


def main():
    """Generate the requested figures and record their output filenames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=FIGURES)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    responses = load_responses()
    human = human_distributions()
    aggregated = aggregate(responses)
    errors = error_table(aggregated, human)
    calibration_figures = {
        "peak_distributions",
        "q66_histograms_ghana",
        "temperature_all_curves",
    }
    curves = calibrated = None
    if calibration_figures.intersection(args.figures):
        curves, calibrated, _ = calibration(aggregated, human)
    report_path = args.output / "generation_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    report = {name: result for name, result in report.items() if name in FIGURES}
    for name in args.figures:
        print("Generating", name, flush=True)
        if name == "heatmap_mae":
            plot_error_heatmap(errors, args.output)
        elif name == "peak_distributions":
            plot_variance(variance_aggregate(responses), human, calibrated, args.output)
        elif name == "temperature_all_curves":
            plot_temperature_curves(curves, args.output)
        elif name == "q66_histograms_ghana":
            plot_ghana_distributions(aggregated, human, calibrated, args.output)
        elif name == "guidance_comparison":
            plot_guidance_comparison(args.output, human)
        elif name == "error_comparison":
            plot_error_comparison(errors, args.output)
        report[name] = {"status": "generated", "filename": name + ".pdf"}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
