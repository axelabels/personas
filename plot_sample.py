"""Plot collected answer probabilities in one grid of question subplots."""
import argparse
import math
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from analysis import QUESTIONS, load_responses, values


def plot_sample(input_path, output_path):
    """Average recorded probability vectors within each question."""
    responses = load_responses(input_path)
    if responses.empty:
        raise ValueError("The response file contains no rows")
    questions = sorted(responses.qkey.unique(), key=lambda question: int(question[1:]))
    columns = min(3, len(questions))
    rows = math.ceil(len(questions) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False,
        layout="constrained",
    )
    for axis, question in zip(axes.flat, questions):
        question_responses = responses.loc[responses.qkey.eq(question)]
        probabilities = np.stack(question_responses.probabilities).mean(axis=0)
        answer_values = values(question).astype(int)
        specification = QUESTIONS[question]
        labels = (specification.get("labels") or {})
        axis.bar(answer_values, probabilities, color="#378ADD", width=0.75)
        axis.set_xticks(answer_values, [
            fill(f"{answer}: {labels[str(answer)]}", 17)
            if str(answer) in labels else str(answer)
            for answer in answer_values
        ], fontsize=8)
        axis.set_title(
            fill(f"{question}: {specification['text']}", 48)
            + f"\n(n = {len(question_responses)})",
            fontsize=10,
        )
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylabel("Probability")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(questions):]:
        axis.set_visible(False)
    figure.suptitle(
        "Collected response distributions\n"
        "Mean probabilities normalized over returned numeric tokens",
        fontsize=14,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved {len(questions)} question subplots to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/sample_distributions.png"))
    args = parser.parse_args()
    plot_sample(args.input, args.output)


if __name__ == "__main__":
    main()
