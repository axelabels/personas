"""Rebuild local survey-derived inputs from official WVS and EVS SPSS files."""
import argparse
import ast
import json
import tempfile
from pathlib import Path

import pandas as pd
import pycountry
import pyreadstat

from analysis import DATA, QUESTIONS
from demographics import sociodemographic_persona
from prompts import AUTONOMY, build_survey_prompt, fewshot_persona, value_persona

SURVEY_COLUMNS = json.loads((DATA / "survey_columns.json").read_text(encoding="utf-8"))
DEMOGRAPHIC_COLUMNS = [
    "X003", "X001", "X025", "X025A_01", "X025A", "X025R", "X025CSWVS",
    "X025CSEVS", "X028", "X047R_WVS", "X047R_EVS", "X045", "X007",
    "X013", "X050C",
]


def country_name(code):
    country = pycountry.countries.get(numeric=f"{int(code):03d}")
    return country.name if country else None


def load_official_file(path, study_number):
    """Read numeric codes and SPSS missing values, never formatted value labels."""
    _, metadata = pyreadstat.read_sav(path, metadataonly=True, encoding="latin1")
    expected_rows = {4001: 442473, 7503: 224434}[study_number]
    if metadata.number_rows != expected_rows:
        raise ValueError(f"{path} has an unexpected row count; use the dataset versions in README.md")
    required = set(SURVEY_COLUMNS.values()) | set(DEMOGRAPHIC_COLUMNS)
    required |= {"S007", "S003", "S009", "S020", "S017", "STUDYNO"}
    columns = [name for name in metadata.column_names if name.upper() in required]
    frame, _ = pyreadstat.read_sav(path, usecols=columns, encoding="latin1")
    frame.columns = frame.columns.str.upper()
    if not frame.STUDYNO.eq(study_number).all():
        raise ValueError(f"{path} is not the expected study {study_number}")
    if not frame.S007.is_unique:
        raise ValueError(f"Duplicate S007 identifiers in {path}")
    return frame


def load_surveys(wvs_path, evs_path):
    frame = pd.concat([
        load_official_file(wvs_path, 4001),
        load_official_file(evs_path, 7503),
    ], ignore_index=True)
    frame = frame.sort_values(["S009", "S007"], kind="stable").reset_index(drop=True)
    frame["Country"] = frame.S003.map({code: country_name(code) for code in frame.S003.unique()})
    missing = set(SURVEY_COLUMNS.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Official files lack required survey columns: {sorted(missing)}")
    return frame


def select_respondents(surveys):
    """Recover the historical sample by identifiers, not by stored answers."""
    selection = pd.read_csv(DATA / "respondent_selection.csv")
    sample = selection.merge(
        surveys, left_on=["studyno", "S007"], right_on=["STUDYNO", "S007"],
        how="left", validate="one_to_one", indicator=True,
    )
    if not sample._merge.eq("both").all():
        raise ValueError("Some study respondents are absent; use the documented dataset versions")
    # The original sample filled missing child-quality selections with zero.
    autonomy_columns = [SURVEY_COLUMNS[question] for question in AUTONOMY.values()]
    sample[autonomy_columns] = sample[autonomy_columns].fillna(0)
    return sample.drop(columns="_merge")


def write_human_counts(sample, output):
    records = []
    for country, group in sample.groupby("Country"):
        for question in QUESTIONS:
            if question in AUTONOMY.values() and question not in {"Q8", "Q11", "Q15", "Q17"}:
                continue
            for answer, count in group[SURVEY_COLUMNS[question]].dropna().value_counts().items():
                records.append(dict(country=country, question=question, value=int(answer), count=int(count)))
    pd.DataFrame(records).to_csv(output / "human_counts.csv", index=False)


def write_prompts(sample, output):
    conditions = pd.read_csv(DATA / "prompt_conditions.csv", keep_default_na=False)
    people = sample.dropna(subset="person_id").set_index("person_id")
    systems, users = [], []
    system_ids, user_ids = {}, {}
    index = conditions.copy()
    index["system_id"] = -1
    index["user_id"] = -1
    for row in conditions.itertuples():
        country_label = {"United Kingdom": "the United Kingdom", "Moldova, Republic of": "Moldova", "Taiwan, Province of China": "Taiwan"}.get(row.Country, row.Country)
        country = country_label if row.include_nationality else None
        if row.person_id == -1:
            messages = build_survey_prompt(row.qkey)
        elif row.person_id < 0:
            messages = build_survey_prompt(row.qkey, "country", country_label)
        else:
            person = people.loc[row.person_id]
            answers = {
                question: int(person[column]) for question, column in SURVEY_COLUMNS.items()
                if pd.notna(person[column])
            }
            excluded = ast.literal_eval(row.exclusion_group)
            if row.fewshot == "descriptor":
                persona = value_persona(answers, row.source, excluded)
                messages = build_survey_prompt(row.qkey, "value", country, persona)
            elif row.fewshot == "fewshot":
                examples = fewshot_persona(answers, row.source, excluded)
                messages = build_survey_prompt(row.qkey, "fewshot", country, fewshot_responses=examples)
            elif row.fewshot == "sociodemographic":
                persona = sociodemographic_persona(person.to_dict())
                messages = build_survey_prompt(row.qkey, "sociodemographic", country, persona)
            else:
                raise ValueError(f"Unknown prompt method: {row.fewshot}")
        for message, texts, lookup, column in [
            (messages[0]["content"], systems, system_ids, "system_id"),
            (messages[1]["content"], users, user_ids, "user_id"),
        ]:
            if message not in lookup:
                lookup[message] = len(texts)
                texts.append(message)
            index.loc[row.Index, column] = lookup[message]
    index.to_parquet(output / "prompt_index.parquet", index=False)
    (output / "prompt_text.json").write_text(
        json.dumps(dict(system=systems, user=users), indent=2, ensure_ascii=False), encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wvs", type=Path, required=True, help="Official WVS Trends v4.1 SPSS file")
    parser.add_argument("--evs", type=Path, required=True, help="Official EVS ZA7503 v3.0.0 SPSS file")
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    print("Reading official survey files...", flush=True)
    surveys = load_surveys(args.wvs, args.evs)
    sample = select_respondents(surveys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Finish all computation before replacing any existing local outputs.
    with tempfile.TemporaryDirectory(prefix="survey-preparation-", dir=args.output.parent) as directory:
        staging = Path(directory)
        print("Building human reference counts and persona prompts...", flush=True)
        write_human_counts(sample, staging)
        write_prompts(sample, staging)
        args.output.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            path.replace(args.output / path.name)
    print(f"Prepared local survey-derived files in {args.output}", flush=True)


if __name__ == "__main__":
    main()
