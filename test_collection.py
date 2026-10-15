"""Offline checks for the collection example; API calls are replaced with mocks."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

import analysis
import collect_responses as collector
import prompts


def prompt_fixture():
    """Synthetic generic prompt; no official survey records are needed."""
    return {
        "prompt_id": 0,
        "metadata": dict(
            Country="none", person_id=-1, qkey="Q1", include_nationality=False,
            fewshot="descriptor", source="composable_claude_sonnet_4_6",
            exclusion_group="()",
        ),
        "question": "Q1",
        "messages": prompts.build_survey_prompt("Q1"),
    }


def completion_fixture():
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "1", "refusal": None},
            "logprobs": {"content": [{
                "token": "1", "logprob": math.log(0.4),
                "top_logprobs": [
                    {"token": "1", "logprob": math.log(0.4)},
                    {"token": " 1", "logprob": math.log(0.1)},
                    {"token": "2", "logprob": math.log(0.3)},
                    {"token": "word", "logprob": math.log(0.2)},
                ],
            }]},
        }],
    }


class CollectionChecks(unittest.TestCase):
    def test_numeric_probabilities_and_duplicate_sampled_token(self):
        logprobs = collector.answer_logprobs(completion_fixture(), "Q1")
        np.testing.assert_allclose(
            analysis.decode_logprobs(json.dumps(logprobs), "Q1"),
            [0.625, 0.375, 0, 0],
        )

    def test_ten_point_tokens_keep_original_encoding(self):
        response = completion_fixture()
        token = response["choices"][0]["logprobs"]["content"][0]
        response["choices"][0]["message"]["content"] = "0"
        token.update(token="0", logprob=math.log(0.5), top_logprobs=[
            {"token": "0", "logprob": math.log(0.5)},
            {"token": "9", "logprob": math.log(0.5)},
        ])
        logprobs = collector.answer_logprobs(response, "Q49")
        np.testing.assert_allclose(
            analysis.decode_logprobs(json.dumps(logprobs), "Q49"),
            [0.5] + [0] * 8 + [0.5],
        )

    def test_reject_unusable_answers(self):
        for failure in ["no_logprobs", "explanation", "leading_token", "truncated", "refusal"]:
            with self.subTest(failure=failure):
                response = completion_fixture()
                choice = response["choices"][0]
                if failure == "no_logprobs":
                    choice["logprobs"] = None
                elif failure == "explanation":
                    choice["message"]["content"] = "I choose 1"
                elif failure == "leading_token":
                    choice["logprobs"]["content"][0]["token"] = "\n"
                elif failure == "truncated":
                    choice["finish_reason"] = "length"
                else:
                    choice["message"]["refusal"] = "refused"
                with self.assertRaises(ValueError):
                    collector.answer_logprobs(response, "Q1")

    def test_export_collection_and_analysis_roundtrip(self):
        record = prompt_fixture()
        client = Mock()
        client.chat.completions.create.return_value.model_dump.return_value = completion_fixture()
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "prompts.jsonl"
            output_path = Path(directory) / "responses.parquet"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            inputs = collector.read_prompts(input_path, 5)
            count = collector.collect_responses(client, inputs, "test-model", output_path)
            self.assertEqual(count, 1)
            received = client.chat.completions.create.call_args.kwargs
            self.assertEqual(received["messages"], record["messages"])
            self.assertTrue(received["logprobs"])
            self.assertEqual(received["top_logprobs"], 5)
            loaded = analysis.load_responses(output_path)
            np.testing.assert_allclose(loaded.iloc[0].probabilities, [0.625, 0.375, 0, 0])
            for field in collector.CONDITION_FIELDS:
                self.assertEqual(loaded.iloc[0][field], record["metadata"][field])
            with self.assertRaises(FileExistsError):
                collector.collect_responses(client, inputs, "test-model", output_path)
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_checkpoint_survives_failed_request(self):
        record = prompt_fixture()
        completed = Mock()
        completed.model_dump.return_value = completion_fixture()
        client = Mock()
        client.chat.completions.create.side_effect = [completed, RuntimeError("API error")]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "responses.parquet"
            with self.assertRaisesRegex(RuntimeError, "API error"):
                collector.collect_responses(client, [record, record], "test-model", output_path)
            self.assertEqual(len(pd.read_parquet(output_path)), 1)
            self.assertEqual(len(output_path.with_suffix(".raw.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
