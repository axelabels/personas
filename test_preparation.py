"""Checks for local data preparation and public packaging; no API calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

import build_release
from demographics import sociodemographic_persona
import prepare_data
from prompts import build_survey_prompt, fewshot_persona, value_persona


class PreparationChecks(unittest.TestCase):
    def test_selection_requires_every_historical_respondent(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            pd.DataFrame([dict(studyno=4001, S007=123, person_id=0)]).to_csv(
                data / "respondent_selection.csv", index=False,
            )
            surveys = pd.DataFrame([dict(STUDYNO=4001, S007=999)])
            with patch.object(prepare_data, "DATA", data):
                with self.assertRaisesRegex(ValueError, "respondents are absent"):
                    prepare_data.select_respondents(surveys)

    def test_prompt_methods_are_reconstructed_from_supplied_answers(self):
        answers = {"Q164": 1, "Q209": 2, "Q8": 1}
        person = {column: np.nan for column in prepare_data.SURVEY_COLUMNS.values()}
        person.update({prepare_data.SURVEY_COLUMNS[key]: value for key, value in answers.items()})
        person.update(person_id=0, X003=30, X001=2)
        sample = pd.DataFrame([person])
        conditions = pd.DataFrame([
            dict(prompt_id=index, Country="Ghana", person_id=0, qkey="Q1",
                 include_nationality=True, fewshot=method,
                 source="composable_claude_sonnet_4_6", exclusion_group="()")
            for index, method in enumerate(["descriptor", "fewshot", "sociodemographic"])
        ])
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            conditions.to_csv(data / "prompt_conditions.csv", index=False)
            with patch.object(prepare_data, "DATA", data):
                prepare_data.write_prompts(sample, data)
            text = json.loads((data / "prompt_text.json").read_text(encoding="utf-8"))
            index = pd.read_parquet(data / "prompt_index.parquet")
            expected = [
                build_survey_prompt("Q1", "value", "Ghana", value_persona(answers)),
                build_survey_prompt("Q1", "fewshot", "Ghana", fewshot_responses=fewshot_persona(answers)),
                build_survey_prompt("Q1", "sociodemographic", "Ghana", "A 30 years old female."),
            ]
            for row, messages in zip(index.itertuples(), expected):
                self.assertEqual(text["system"][row.system_id], messages[0]["content"])
                self.assertEqual(text["user"][row.user_id], messages[1]["content"])

    def test_evs_income_fallback_and_historical_demographic_fields(self):
        description = sociodemographic_persona({
            "X003": 40, "X001": 1, "X047R_WVS": np.nan, "X047R_EVS": 2,
            "X011": 3, "X013": 4,
        })
        self.assertEqual(description, "A 40 years old male, in a lower-middle income household, in a 4-person household.")

    def test_public_archive_excludes_generated_survey_files(self):
        forbidden = {
            "data/prompt_text.json", "data/prompt_index.parquet", "data/human_counts.csv",
        }
        self.assertFalse(forbidden.intersection(build_release.PUBLIC_FILES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.zip"
            build_release.build_release(path)
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                self.assertFalse({f"public_release/{name}" for name in forbidden}.intersection(names))
                self.assertEqual(sum(name.endswith(".pdf") for name in names), 6)


if __name__ == "__main__":
    unittest.main()
