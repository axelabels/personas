"""Build paper prompt variants or export the exact archived messages.

No API clients, credentials, response generation, or network operations are used.
Template reconstruction is distinct from an exact archived prompt; see README.md.
"""
import argparse
import json
from pathlib import Path
import re

from analysis import require_prepared_data

DATA = Path(__file__).resolve().parent / "data"
STUDY = json.loads((DATA / "study.json").read_text(encoding="utf-8"))
QUESTIONS = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
GUIDANCE = json.loads((DATA / "guidance_templates.json").read_text(encoding="utf-8"))
DESCRIPTORS = json.loads((DATA / "descriptors.json").read_text(encoding="utf-8"))
AUTONOMY = {
    "good manners": "Q7",
    "independence": "Q8",
    "hard work": "Q9",
    "feeling of responsibility": "Q10",
    "imagination": "Q11",
    "tolerance and respect for other people": "Q12",
    "thrift": "Q13",
    "determination and perseverance": "Q14",
    "religious faith": "Q15",
    "not being selfish": "Q16",
    "obedience": "Q17",
}
DESCRIPTOR_INSTRUCTIONS = {
    "composable": 'For every response option, craft a one-sentence persona snippet starting with "You." Each snippet should nudge an LLM toward giving that specific answer while suppressing broader inferences the model might otherwise make. The snippets must be self-contained and composable—mixing and matching one per question should produce a coherent, accurate persona.',
    "modular": 'For each possible response, provide a short sentence describing the person who gave that answer. The descriptions should be modular, such that one descriptor per question can be combined into a coherent role-play persona.\n\nEach descriptor should trigger the target response while hedging against broader associations the LLM might naturally infer. For example, opposition to homosexuality should not necessarily imply broader conservative views, and reporting that one has voted should not necessarily imply strong political engagement.\n\nStart each descriptor with "You."',
    "no_hedge": 'For each possible response, provide a short sentence describing the person who gave that answer. The descriptions should be composable, such that one descriptor per question can be combined into a coherent role-play persona.\n\nEach descriptor should trigger the target response.\n\nStart each descriptor with "You."',
}


def user_message(question, inverted=False):
    """Preserve the original code's text shift and its unshifted Scale labels."""
    if question not in QUESTIONS:
        raise ValueError("Question is outside the paper scope")
    question_spec = QUESTIONS[question]
    if inverted:
        question_spec = json.loads(
            (DATA / "inverted_questions.json").read_text(encoding="utf-8")
        )[question]
    if "autonomy_trait" in question_spec:
        return (
            question_spec["text"]
            + "\n\nList the qualities you consider especially important, one per line."
        )
    minimum, maximum = (question_spec["min"], question_spec["max"])
    text = question_spec["text"]
    if (minimum, maximum) == (1, 10):
        text = re.sub("\\b1\\b", "0", re.sub("\\b10\\b", "9", text))
        minimum, maximum = (0, 9)
    labels = question_spec.get("labels")
    scale = (
        ""
        if not labels
        else "Scale: "
        + ", ".join((f"{answer}={label}" for answer, label in labels.items()))
        + "\n"
    )
    return (
        f"{text}\n\n{scale}Respond with ONLY the single integer ({minimum}–{maximum}):"
    )


def value_persona(answers, source="composable_claude_sonnet_4_6", excluded=()):
    """Convert caller-supplied descriptor-question answers into the saved snippets."""
    if source not in DESCRIPTORS:
        raise ValueError("Descriptor source is outside the paper scope")
    descriptor_table = DESCRIPTORS[source]
    question_descriptors = {}
    for question_key, entries in descriptor_table.items():
        if question_key == "autonomy":
            for trait, text in entries.items():
                trait_question = AUTONOMY.get(trait.replace("_", " "))
                if trait_question:
                    question_descriptors[trait_question] = {"1": text}
        elif question_key in QUESTIONS:
            question_descriptors[question_key] = entries
    pieces = []
    for question_key in sorted(
        question_descriptors, key=lambda question_key: int(question_key[1:])
    ):
        if question_key in excluded or question_key not in answers:
            continue
        answer = str(int(answers[question_key]))
        text = question_descriptors[question_key].get(answer)
        if text:
            pieces.append(text.strip())
    if not pieces:
        raise ValueError("No valid descriptor answers supplied")
    return " ".join(pieces).replace("---", ",")


def fewshot_persona(answers, source="composable_claude_sonnet_4_6", excluded=()):
    """Original fewshot block, with autonomy qualities and numbered responses."""
    if source not in DESCRIPTORS:
        raise ValueError("Unknown descriptor source")
    blocks = []
    traits = [
        trait
        for trait, question_key in AUTONOMY.items()
        if answers.get(question_key) == 1
    ]
    if traits:
        blocks.append(
            "Question:\n" + QUESTIONS["Q8"]["text"] + "\nAnswer: " + ", ".join(traits)
        )
    for question_key in sorted(
        [
            question_key
            for question_key in DESCRIPTORS[source]
            if question_key != "autonomy"
        ],
        key=lambda question_key: int(question_key[1:]),
    ):
        if question_key in excluded or question_key not in answers:
            continue
        question = QUESTIONS[question_key]
        labels = question.get("labels")
        text = question["text"]
        scale = (
            ""
            if not labels
            else "\nScale: "
            + ", ".join((f"{answer}={label}" for answer, label in labels.items()))
        )
        blocks.append(f"Question:\n{text}{scale}\nAnswer: {int(answers[question_key])}")
    return "Persona responses: \n\n" + "\n\n".join(blocks)


def build_survey_prompt(
    question, method="generic", country=None, persona=None, fewshot_responses=None
):
    """Full messages for the main study's social-science condition.

    Supply an existing sociodemographic persona as text. Value personas can be
    constructed with value_persona(); fewshot_responses is the recorded block.
    """
    if question not in STUDY["target_questions"]:
        raise ValueError("Not a paper target question")
    if method not in ["generic", "country", "value", "sociodemographic", "fewshot"]:
        raise ValueError("Unknown method")
    if method == "generic":
        system = "You are participating in a social science simulation. You will be given a survey question and must answer it as a typical person would, embracing this person's values as your own.\n"
    elif method == "country":
        if not country:
            raise ValueError("Country required")
        system = f"You are participating in a social science simulation. You will be given a survey question and must answer it as someone living in {country} would, embracing this person's values as your own.\n"
    else:
        system = (
            GUIDANCE["persona"]["social_science"].replace("{country}", country)
            if country
            else GUIDANCE["persona_no_country"]["social_science"]
        ) + "\n"
        if method == "fewshot":
            if not fewshot_responses:
                raise ValueError("Recorded fewshot responses required")
            system += (
                "To help you, here are some example responses given by the persona you have to emulate.\n\n"
                + fewshot_responses
                + "\n"
            )
        else:
            if not persona:
                raise ValueError("Persona text required")
            system += "Your persona: " + persona + "\n"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(question)},
    ]


def build_guidance_prompt(question, country, guidance="social_science", inverted=False):
    if question in STUDY["target_questions"]:
        raise ValueError("Guidance comparison uses descriptor questions only")
    if guidance not in STUDY["guidance_variants"]:
        raise ValueError(f"Unknown guidance variant: {guidance}")
    system = GUIDANCE["country"][guidance].format(country=country)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(question, inverted)},
    ]


def descriptor_generation_prompt(question, variant="composable"):
    """Paper Table's generator instruction plus the relevant original question.

    Compose the manuscript instruction and question into a user message.
    """
    if question in STUDY["target_questions"] or question not in QUESTIONS:
        raise ValueError("Expected a descriptor question")
    question_spec = QUESTIONS[question]
    labels = question_spec.get("labels")
    scale = (
        ""
        if not labels
        else "\nScale: "
        + ", ".join((f"{answer}={label}" for answer, label in labels.items()))
    )
    return [
        {
            "role": "user",
            "content": DESCRIPTOR_INSTRUCTIONS[variant]
            + "\n\nQuestion:\n"
            + question_spec["text"]
            + scale,
        }
    ]


def archived_prompts():
    import pandas as pd

    require_prepared_data("prompt_index.parquet", "prompt_text.json")
    index = pd.read_parquet(DATA / "prompt_index.parquet")
    prompt_text = json.loads((DATA / "prompt_text.json").read_text(encoding="utf-8"))
    for row in index.itertuples(index=False):
        yield {
            "prompt_id": int(row.prompt_id),
            "metadata": {
                "Country": row.Country,
                "person_id": int(row.person_id),
                "qkey": row.qkey,
                "include_nationality": bool(row.include_nationality),
                "fewshot": row.fewshot,
                "source": row.source,
                "exclusion_group": row.exclusion_group,
            },
            "question": row.qkey,
            "method": row.fewshot,
            "country": row.Country,
            "messages": [
                {"role": "system", "content": prompt_text["system"][row.system_id]},
                {"role": "user", "content": prompt_text["user"][row.user_id]},
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-id",
        type=int,
        help="Export one archived prompt; omit to export all archived prompts",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in archived_prompts():
            if args.prompt_id is not None and record["prompt_id"] != args.prompt_id:
                continue
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if not count:
        raise SystemExit("No matching archived prompt")
    print(f"Exported {count} archived prompts to {args.output}")


if __name__ == "__main__":
    main()
