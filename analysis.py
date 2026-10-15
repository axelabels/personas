"""Probability decoding, aggregation, and calibration for the study figures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
STUDY = json.loads((DATA / "study.json").read_text(encoding="utf-8"))
QUESTIONS = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
MODELS = STUDY["models"]
COUNTRIES = STUDY["countries"]
TARGETS = STUDY["target_questions"]
METHODS = [
    "Generic prompt",
    "Country prompt",
    "Sociodemographic prompt",
    "Value prompt",
]
COLORS = dict(zip(METHODS, ["#378ADD", "#1D9E75", "#9B59B6", "#FF5733"]))
TEMPERATURES = np.logspace(-1, 2, 31)
SMOOTHING_MASS = 1e-3
MEAN_TOLERANCE = 1e-6


def clean_model_name(name):
    """Shorten model identifiers for plot labels."""
    for suffix in [
        "-logprobs",
        "-preview",
        "-A3B-FP8",
        "-IT-NVFP4",
        "-Instruct-2512",
        "-it",
    ]:
        name = name.replace(suffix, "")
    return name


def scale(question):
    """Return the response range used to normalize errors."""
    return QUESTIONS[question]["max"] - QUESTIONS[question]["min"]


def values(question):
    """Return valid survey response values in probability-vector order."""
    return np.arange(
        QUESTIONS[question]["min"], QUESTIONS[question]["max"] + 1, dtype=float
    )


def decode_logprobs(raw, question):
    """Normalize admissible tokens; ten-point tokens 0–9 represent answers 1–10."""
    token_logprobs = json.loads(raw)
    response_values = values(question)
    token_values = (
        range(10) if len(response_values) == 10 else response_values.astype(int)
    )
    log_probabilities = np.array(
        [float(token_logprobs.get(str(token), -np.inf)) for token in token_values]
    )
    if not np.isfinite(log_probabilities).any():
        raise ValueError(f"No valid answer probabilities for {question}: {raw}")
    probabilities = np.exp(log_probabilities - np.max(log_probabilities))
    return probabilities / probabilities.sum()


def load_responses(path=DATA / "responses.parquet"):
    """Load token probabilities and derive each response's expected value."""
    responses = pd.read_parquet(path)
    responses["probabilities"] = [
        decode_logprobs(raw, question)
        for raw, question in zip(responses.logprobs, responses.qkey)
    ]
    responses["mean"] = [
        probabilities @ values(question)
        for probabilities, question in zip(responses.probabilities, responses.qkey)
    ]
    return responses


def require_prepared_data(*filenames):
    """Explain how to create local survey inputs when they are not present."""
    missing = [name for name in filenames if not (DATA / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Local survey data have not been prepared: " + ", ".join(missing)
            + ". Run prepare_data.py with the official WVS and EVS files; see README.md."
        )


def human_distributions():
    """Normalize survey counts for each country and question."""
    require_prepared_data("human_counts.csv")
    counts = pd.read_csv(DATA / "human_counts.csv")
    distributions = {}
    for (country, question), group in counts.groupby(["country", "question"]):
        frequencies = (
            group.set_index("value")["count"]
            .reindex(values(question), fill_value=0)
            .to_numpy(dtype=float)
        )
        distributions[country, question] = frequencies / frequencies.sum()
    return distributions


def method_rows(responses, method, nationality=True):
    """Select a prompting condition using the recorded persona identifiers."""
    selected = responses.exclusion_group.eq("()")
    # Negative IDs encode baseline conditions; nonnegative IDs identify personas.
    if method == "Generic prompt":
        return responses[selected & (responses.person_id == -1)]
    if method == "Country prompt":
        return responses[
            selected & (responses.person_id < -2) & (responses.person_id > -100)
        ]
    selected &= (responses.person_id >= 0) & responses.include_nationality.eq(
        nationality
    )
    condition = {
        "Value prompt": "descriptor",
        "Sociodemographic prompt": "sociodemographic",
    }[method]
    selected &= responses.fewshot.eq(condition)
    if method == "Value prompt":
        selected &= responses.source.eq("composable_claude_sonnet_4_6")
    return responses[selected]


def aggregate(responses):
    """Average persona distributions before computing errors or variances."""
    records = []
    for method in METHODS:
        selected = method_rows(responses, method)
        for (model, country, question), group in selected.groupby(
            ["model", "Country", "qkey"], observed=True
        ):
            probabilities = np.stack(group.probabilities).mean(axis=0)
            # A generic prompt has no country, so evaluate it against each reference.
            evaluation_countries = (
                COUNTRIES if method == "Generic prompt" else [country]
            )
            for evaluation_country in evaluation_countries:
                records.append(
                    dict(
                        model=model,
                        country=evaluation_country,
                        question=question,
                        method=method,
                        probabilities=probabilities,
                    )
                )
    return pd.DataFrame(records)


def variance_aggregate(responses):
    """Use the same persona/question cases across all models in variance plots."""
    keys = [
        "Country",
        "person_id",
        "qkey",
        "include_nationality",
        "fewshot",
        "source",
        "exclusion_group",
    ]
    counts = (
        responses.groupby(keys, observed=True)
        .model.nunique()
        .rename("model_count")
        .reset_index()
    )
    shared_keys = counts[counts.model_count == len(MODELS)][keys]
    shared_responses = responses.merge(shared_keys, on=keys, validate="many_to_one")
    return aggregate(shared_responses)


def error_table(aggregated, human):
    """Compute normalized absolute error of each aggregate expected response."""
    records = []
    for response in aggregated.itertuples(index=False):
        reference_key = (response.country, response.question)
        if reference_key not in human:
            continue
        error = abs(
            (response.probabilities - human[reference_key]) @ values(response.question)
        ) / scale(response.question)
        records.append(
            dict(
                model=response.model,
                country=response.country,
                question=response.question,
                method=response.method,
                error=error,
            )
        )
    return pd.DataFrame(records)


def temperature_scale(probabilities, temperature, preserve_mean=False):
    """Scale categorical probabilities, optionally preserving their expected value.

    Unit temperature is an exact identity. Other temperatures use the study's
    additive smoothing. Bisection finds an exponential tilt that restores the mean.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if np.isclose(temperature, 1, rtol=0, atol=1e-14):
        return probabilities.copy()
    levels = np.arange(probabilities.shape[-1], dtype=float)
    smoothed = probabilities + SMOOTHING_MASS
    log_weights = np.log(smoothed / smoothed.sum(axis=-1, keepdims=True)) / temperature

    def normalized(tilt):
        scores = log_weights + tilt[..., None] * levels
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        return weights / weights.sum(axis=-1, keepdims=True)

    if not preserve_mean:
        return normalized(np.zeros(probabilities.shape[:-1]))
    target_mean = probabilities @ levels
    lower = np.full_like(target_mean, -100.0)
    upper = np.full_like(target_mean, 100.0)
    result = np.zeros_like(probabilities)
    converged = np.zeros_like(target_mean, dtype=bool)
    for _ in range(200):
        midpoint = (lower + upper) / 2
        candidate = normalized(midpoint)
        candidate_mean = candidate @ levels
        newly_converged = (~converged) & (
            np.abs(candidate_mean - target_mean) < MEAN_TOLERANCE
        )
        result[newly_converged] = candidate[newly_converged]
        converged |= newly_converged
        if converged.all():
            break
        lower = np.where((~converged) & (candidate_mean < target_mean), midpoint, lower)
        upper = np.where(
            (~converged) & (candidate_mean >= target_mean), midpoint, upper
        )
    result[~converged] = candidate[~converged]
    return result


def normalized_variance(probabilities):
    """Divide categorical variance by the maximum variance on the same scale."""
    levels = np.arange(len(probabilities))
    mean = probabilities @ levels
    return float(
        probabilities @ ((levels - mean) ** 2) / (((len(probabilities) - 1) / 2) ** 2)
    )


def calibration(aggregated, human):
    """Select temperatures with leave-one-(question, country)-out validation.

    For each held-out pair, minimize mean Wasserstein distance on every other
    pair. Other countries' responses to the same question remain in training.
    """
    records, calibrated, chosen = [], {}, {}
    for (model, method), group in aggregated.groupby(["model", "method"], sort=False):
        group = group[
            group.apply(lambda row: (row.country, row.question) in human, axis=1)
        ].reset_index(drop=True)
        for mode in ["Classic", "Mean-preserving"]:
            losses = np.zeros((len(TEMPERATURES), len(group)))
            scaled_distributions = {}
            for temperature_index, temperature in enumerate(TEMPERATURES):
                # Batch items by scale length so probability arrays remain rectangular.
                for category_count, batch in group.groupby(
                    group.question.map(lambda question: len(values(question)))
                ):
                    row_indices = batch.index.to_numpy()
                    probabilities = np.stack(batch.probabilities)
                    references = np.stack(
                        [
                            human[country, question]
                            for country, question in zip(batch.country, batch.question)
                        ]
                    )
                    scaled = temperature_scale(
                        probabilities, temperature, mode == "Mean-preserving"
                    )
                    wasserstein = np.abs(
                        np.cumsum(scaled - references, axis=1)[:, :-1]
                    ).sum(axis=1) / (category_count - 1)
                    absolute_errors = np.abs(
                        (scaled - references) @ np.arange(category_count)
                    ) / (category_count - 1)
                    losses[temperature_index, row_indices] = wasserstein
                    for row_index, distribution, error, distance in zip(
                        row_indices, scaled, absolute_errors, wasserstein
                    ):
                        row = group.loc[row_index]
                        records.append(
                            dict(
                                model=model,
                                method=method,
                                country=row.country,
                                question=row.question,
                                temperature=temperature,
                                temp_method=mode,
                                mae=error,
                                wasserstein=distance,
                            )
                        )
                        if mode == "Mean-preserving":
                            scaled_distributions[
                                temperature_index, row_index
                            ] = distribution
            if mode == "Mean-preserving":
                if len(group) < 2:
                    raise ValueError(
                        "Calibration requires at least two question-country pairs"
                    )
                training_losses = (losses.sum(axis=1)[:, None] - losses) / (
                    len(group) - 1
                )
                best_indices = training_losses.argmin(axis=0)
                for row_index, temperature_index in enumerate(best_indices):
                    row = group.loc[row_index]
                    key = (model, method, row.country, row.question)
                    calibrated[key] = scaled_distributions[temperature_index, row_index]
                    chosen[key] = float(TEMPERATURES[temperature_index])
    return pd.DataFrame(records), calibrated, chosen
