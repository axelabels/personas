"""Scientific invariants and archive integrity checks; no model calls."""
import json
import unittest
import numpy as np
import pandas as pd
import analysis
import prompts


class ReproductionChecks(unittest.TestCase):
    def test_token_mapping(self):
        np.testing.assert_allclose(
            analysis.decode_logprobs('{"0":0,"9":0,"10":5}', "Q49"), [0.5] + [0.0] * 8 + [0.5]
        )
        np.testing.assert_allclose(
            analysis.decode_logprobs('{"1":0,"2":0,"junk":9}', "Q1"), [0.5, 0.5, 0, 0]
        )
        with self.assertRaises(ValueError):
            analysis.decode_logprobs('{"junk":0}', "Q1")

    def test_calibration_preserves_mean(self):
        rng = np.random.default_rng(7)
        for category_count in [2, 3, 4, 8, 10]:
            probabilities = np.vstack(
                [
                    rng.dirichlet(np.ones(category_count), size=20),
                    np.eye(category_count),
                ]
            )
            for temperature in [0.1, 0.5, 1.0, 5.0, 100.0]:
                scaled = analysis.temperature_scale(probabilities, temperature, True)
                self.assertTrue(np.isfinite(scaled).all())
                self.assertTrue((scaled >= 0).all())
                np.testing.assert_allclose(scaled.sum(axis=1), 1, atol=1e-12)
                np.testing.assert_allclose(
                    scaled @ np.arange(category_count),
                    probabilities @ np.arange(category_count),
                    atol=1.01e-6,
                )
            np.testing.assert_array_equal(
                analysis.temperature_scale(probabilities, 1, True), probabilities
            )

    def test_leave_one_pair_out(self):
        rows = []
        human = {}
        for country, question, probabilities, reference in [
            ("A", "Q1", [0.1, 0.1, 0.1, 0.7], [0.25] * 4),
            ("B", "Q1", [0.7, 0.1, 0.1, 0.1], [0.25] * 4),
            ("A", "Q2", [0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]),
        ]:
            rows.append(
                dict(
                    model="test",
                    method="Value prompt",
                    country=country,
                    question=question,
                    probabilities=np.array(probabilities),
                )
            )
            human[country, question] = np.array(reference)
        curves, _, chosen = analysis.calibration(pd.DataFrame(rows), human)
        for held_out_pair in [("A", "Q1"), ("B", "Q1"), ("A", "Q2")]:
            training_curves = curves[
                (curves.temp_method == "Mean-preserving")
                & ~(
                    (curves.country == held_out_pair[0])
                    & (curves.question == held_out_pair[1])
                )
            ]
            best_temperature = (
                training_curves.groupby("temperature").wasserstein.mean().idxmin()
            )
            self.assertEqual(
                chosen[("test", "Value prompt") + held_out_pair], best_temperature
            )

    def test_archive_links_and_scope(self):
        responses = pd.read_parquet(analysis.DATA / "responses.parquet")
        prompt_index = pd.read_parquet(analysis.DATA / "prompt_index.parquet")
        self.assertEqual(set(responses.model), set(analysis.MODELS))
        self.assertEqual(set(responses.qkey), set(analysis.TARGETS))
        self.assertTrue(set(responses.Country) <= set(analysis.COUNTRIES + ["none"]))
        self.assertTrue(
            set(responses.prompt_id.dropna()) <= set(prompt_index.prompt_id)
        )
        self.assertTrue(prompt_index.prompt_id.is_unique)
        self.assertFalse(responses.logprobs.str.contains("ERROR").any())
        self.assertTrue(
            responses.groupby(
                [
                    "model",
                    "Country",
                    "person_id",
                    "qkey",
                    "include_nationality",
                    "fewshot",
                    "source",
                    "exclusion_group",
                ],
                observed=True,
            )
            .size()
            .eq(1)
            .all()
        )

    def test_exact_prompt_roundtrip(self):
        index = pd.read_parquet(analysis.DATA / "prompt_index.parquet")
        text = json.loads((analysis.DATA / "prompt_text.json").read_text(encoding="utf-8"))
        exported = {
            record["prompt_id"]: record for record in prompts.archived_prompts()
        }
        for record in index.itertuples(index=False):
            self.assertEqual(
                exported[record.prompt_id]["messages"],
                [
                    {"role": "system", "content": text["system"][record.system_id]},
                    {"role": "user", "content": text["user"][record.user_id]},
                ],
            )
        for messages in text.values():
            for message in messages:
                self.assertNotIn("â€™", message)
                self.assertNotIn("â€”", message)
        for templates in prompts.GUIDANCE.values():
            for template in templates.values():
                self.assertNotIn("â€™", template)
                self.assertNotIn("â€”", template)
        # Every main-study archived user message is exactly reproduced by its builder.
        for question, group in index.groupby("qkey", observed=True):
            for user_id in group.user_id.unique():
                self.assertEqual(text["user"][user_id], prompts.user_message(question))

    def test_guidance_and_descriptor_builders(self):
        for name in analysis.STUDY["guidance_variants"]:
            for question in ["Q164", "Q8"]:
                messages = prompts.build_guidance_prompt(question, "Ghana", name)
                self.assertEqual(len(messages), 2)
                self.assertNotIn("{country}", messages[0]["content"])
        for source in analysis.STUDY["descriptor_sources"]:
            persona = prompts.value_persona({"Q164": 1, "Q209": 1, "Q8": 1}, source)
            self.assertIn(prompts.DESCRIPTORS[source]["Q209"]["1"], persona)
            messages = prompts.build_survey_prompt("Q1", "value", "Ghana", persona)
            self.assertIn(persona, messages[0]["content"])
        with self.assertRaises(ValueError):
            prompts.build_survey_prompt("Q999")

    def test_complete_case_selection(self):
        responses = analysis.load_responses()
        result = analysis.variance_aggregate(responses)
        self.assertEqual(set(result.model), set(analysis.MODELS))
        self.assertFalse((result.method == "Sociodemographic prompt").any())


if __name__ == "__main__":
    unittest.main()
