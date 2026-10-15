"""Collect a small set of survey responses through a Chat Completions endpoint.

Install the optional openai package and set OPENAI_API_KEY before running.
The selected model must support logprobs, top_logprobs, and temperature=1.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import QUESTIONS, TARGETS, decode_logprobs

CONDITION_FIELDS = (
    "Country",
    "person_id",
    "qkey",
    "include_nationality",
    "fewshot",
    "source",
    "exclusion_group",
)


def read_prompts(path, limit):
    """Validate exported messages and metadata before submitting any requests."""
    records = []
    with Path(path).open(encoding="utf-8-sig") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record["metadata"]
            for field in CONDITION_FIELDS:
                if field not in metadata:
                    raise ValueError(f"Prompt metadata must include {field}")
            if metadata["qkey"] not in TARGETS:
                raise ValueError("This example collects main-study survey questions")
            if record.get("question", metadata["qkey"]) != metadata["qkey"]:
                raise ValueError("Question and metadata.qkey must agree")
            messages = record["messages"]
            if not messages or any(
                message.get("role") not in {"system", "user", "assistant", "developer"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError("Messages must contain roles and text content")
            records.append(record)
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError("The input contains no prompts")
    return records


def answer_logprobs(completion, question):
    """Extract numeric alternatives at the first generated token position.

    Requiring the answer at the first position avoids interpreting probabilities
    conditioned on generated explanations or whitespace as answer probabilities.
    """
    choice = completion["choices"][0]
    if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
        raise ValueError("Expected a completed, non-refusal survey answer")
    question_spec = QUESTIONS[question]
    minimum, maximum = question_spec["min"], question_spec["max"]
    valid_tokens = (
        {str(value) for value in range(10)}
        if (minimum, maximum) == (1, 10)
        else {str(value) for value in range(minimum, maximum + 1)}
    )
    content = choice["message"].get("content") or ""
    positions = (choice.get("logprobs") or {}).get("content") or []
    if content.strip() not in valid_tokens or not positions:
        raise ValueError("Expected a single integer with token log-probabilities")
    first_token = positions[0]
    if first_token["token"].strip() != content.strip():
        raise ValueError("The first generated token must contain the integer answer")
    # The sampled token is also usually in top_logprobs; include it only once.
    alternatives = {
        item["token"]: item["logprob"] for item in first_token["top_logprobs"]
    }
    alternatives.setdefault(first_token["token"], first_token["logprob"])
    numeric_logprobs = {}
    for token, logprob in alternatives.items():
        answer = token.strip()
        if answer in valid_tokens and math.isfinite(logprob) and -9999 < logprob <= 0:
            # Different token spellings (for example '1' and ' 1') share an answer.
            numeric_logprobs[answer] = float(
                np.logaddexp(numeric_logprobs.get(answer, -np.inf), logprob)
            )
    if not numeric_logprobs:
        raise ValueError("No admissible numeric token probabilities were returned")
    return numeric_logprobs


def collect_responses(
    client, prompts, model, output, max_tokens=32, reasoning_effort=None
):
    """Save each raw response and checkpoint accepted rows after each request."""
    output = Path(output)
    raw_output = output.with_suffix(".raw.jsonl")
    if output.suffix != ".parquet":
        raise ValueError("Use a .parquet output filename")
    if output.exists() or raw_output.exists():
        raise FileExistsError("Choose a new output filename for each collection run")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with raw_output.open("x", encoding="utf-8") as raw_file:
        for prompt in prompts:
            request = dict(
                model=model,
                messages=prompt["messages"],
                temperature=1,
                logprobs=True,
                top_logprobs=5,
                max_completion_tokens=max_tokens,
            )
            if reasoning_effort is not None:
                request["reasoning_effort"] = reasoning_effort
            completion = client.chat.completions.create(**request).model_dump(
                mode="json"
            )
            raw_file.write(
                json.dumps(
                    {"prompt": prompt, "request": request, "response": completion}
                )
                + "\n"
            )
            raw_file.flush()
            metadata = {field: prompt["metadata"][field] for field in CONDITION_FIELDS}
            logprobs = answer_logprobs(completion, metadata["qkey"])
            encoded_logprobs = json.dumps(logprobs)
            decode_logprobs(encoded_logprobs, metadata["qkey"])
            rows.append(
                dict(
                    model=model,
                    **metadata,
                    prompt_id=prompt.get("prompt_id"),
                    logprobs=encoded_logprobs,
                )
            )
            pd.DataFrame(rows).to_parquet(output, index=False)
            print(f"Saved {len(rows)}/{len(prompts)} responses", flush=True)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", help="Optional OpenAI-compatible endpoint URL")
    parser.add_argument(
        "--limit", type=int, default=0, help="Prompt count; default: all (0)"
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--reasoning-effort", help="Optional model-supported reasoning setting"
    )
    args = parser.parse_args()
    if args.limit < 0 or args.max_tokens < 1:
        parser.error("Use a nonnegative limit and a positive token budget")
    prompts = read_prompts(args.prompts, args.limit)
    # Keep the API dependency optional for offline figure reproduction.
    from openai import OpenAI

    with OpenAI(base_url=args.base_url, timeout=60, max_retries=2) as client:
        collect_responses(
            client,
            prompts,
            args.model,
            args.output,
            args.max_tokens,
            args.reasoning_effort,
        )


if __name__ == "__main__":
    main()
