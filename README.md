# Calibrated value personas

If you use this code or data, please cite the forthcoming [VALE 2026](https://vale2026.iiia.csic.es/) proceedings paper ([preprint](https://arxiv.org/abs/2605.16193)):

```bibtex
@inproceedings{abels2026improvingcrossculturalsurveysimulation,
  title     = {Improving Cross-Cultural Survey Simulation with Calibrated Value Personas},
  author    = {Abels, Axel and Fernandez Domingos, Elias and Shah, Apurva and Lenaerts, Tom},
  booktitle = {Proceedings of the Fourth International Workshop on Value Engineering in AI (VALE 2026)},
  year      = {2026},
  note      = {To appear}
}
```

Offline code and data for the calibration and survey-response analyses in *Improving Cross-Cultural Survey Simulation with Calibrated Value Personas*. The package contains analysis scripts, recorded LLM responses, paper-figure PDFs, and prompt builders. Human reference distributions and participant-specific prompts are generated locally from official EVS/WVS downloads. After that preparation step, figure reproduction is entirely offline. An optional example script collects new responses through an LLM API.

## Data availability and access

| Material | How it is supplied |
| --- | --- |
| Recorded LLM outputs, including outputs conditioned on participant personas | Included in the release |
| Seven reproduced paper figures | Included as PDFs |
| General prompt templates, question definitions, and descriptor lookup tables for possible answers | Included; these do not contain assembled individual personas |
| Prompt conditions and respondent-selection identifiers | Included to connect model outputs to the original experimental conditions |
| Official WVS and EVS respondent datasets | Download separately from the providers using the links below |
| Individual descriptor, few-shot, and sociodemographic prompts | Reconstructed locally from those downloads |
| Human answer counts | Computed locally from those downloads |

The [WVS conditions of use](https://www.worldvaluessurvey.org/AJDownloadLicense.jsp) prohibit redistribution of its data files. The official survey files must therefore be obtained directly from their providers under the applicable terms. Participant-specific prompts and computed human-reference files are also excluded from the public package and must be reconstructed using the downloaded data files.

Reconstruction of the exact prompts and regeneration of figures against human responses require both official datasets and the preparation procedure described below. Figure reproduction does not require additional LLM calls.

Each main-study response's `prompt_id` links to `data/prompt_conditions.csv`. Preparation creates the matching messages in local `data/prompt_text.json` and their lookup index in `data/prompt_index.parquet`. 

## Setup and usage

Run these Bash commands from the release directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the two official SPSS datasets used in the analysis, accepting each provider's terms, and put the extracted files in `private_data/`:

- [WVS Trends 1981–2022](https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp): `Trends_VS_1981_2022_Spss_v4_1.sav` (the study copy records DOI `10.14281/18241.27`).
- [EVS Trend File 1981–2017, ZA7503 v3.0.0](https://europeanvaluesstudy.eu/surveys/trend-file-1981-2017/data-and-documentation-evs-trend-file/): `ZA7503_v3-0-0.sav`, DOI `10.4232/1.14021`.

Use these versions to reproduce the historical respondent selection. The preparation script reads the official files directly; no separately merged survey file is needed.

```bash
python -m pip install -r requirements-data.txt
python prepare_data.py \
  --wvs private_data/Trends_VS_1981_2022_Spss_v4_1.sav \
  --evs private_data/ZA7503_v3-0-0.sav
python -m unittest test_reproduction test_collection test_preparation
python figures.py
```

`prepare_data.py` rebuilds the human answer counts, all participant-specific prompt types (descriptor, few-shot, and sociodemographic), and the prompt index. It uses the recorded respondent selection, so the resulting prompts align with the saved LLM responses. It does not call an LLM.

Use the same activated environment for the examples below. 

The six reproduced paper figures are included as PDFs in `figures/`. Running `python figures.py` regenerates them. Calibration takes a few minutes on a CPU.

To select an output or change the destination:

```bash
python figures.py --figures heatmap_mae
python figures.py --figures guidance_comparison --output results
```

## Figures

| Output | Description |
| --- | --- |
| `heatmap_mae.pdf` | Prediction error by model, country, and prompting method |
| `peak_distributions.pdf` | Human and model response variance |
| `guidance_comparison.pdf` | Comparison of guidance instructions |
| `error_comparison.pdf` | Value prompting versus country prompting |
| `temperature_all_curves.pdf` | Error across calibration temperatures |
| `q66_histograms_ghana.pdf` | Ghana Q66 distributions before and after calibration |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for calculation conventions and validation.

## Recorded LLM results

The release retains the recorded LLM response files. Results from prompts without participant personas include the generic and country-only rows in `data/responses.parquet` (`person_id < 0`) and the country-only guidance results in `data/guidance_responses.parquet`. These model outputs can be read without downloading the human survey data; comparison with human reference distributions requires local data preparation.

`data/responses.parquet` also includes persona-conditioned LLM outputs (`person_id >= 0`) used in the paper's comparisons. The participant-specific prompts and human survey answers are not distributed; they are reconstructed locally from the official files.

## Prompts

Export exact archived messages as JSON Lines:

```bash
python prompts.py --output exports/prompts.jsonl
python prompts.py --prompt-id 0 --output exports/one_prompt.jsonl
```

`data/prompt_conditions.csv` records the experimental conditions; preparation generates `data/prompt_index.parquet` and `data/prompt_text.json` locally. Non-null `prompt_id` values in the response table reference this index.

Construct messages from the study's templates:

```python
from prompts import (
    build_survey_prompt,
    build_guidance_prompt,
    value_persona,
    fewshot_persona,
    descriptor_generation_prompt,
)

# Illustrative profile; supply the desired survey answers for a full persona.
answers = {"Q8": 1, "Q164": 7, "Q209": 2}
persona = value_persona(answers, source="composable_claude_sonnet_4_6")
messages = build_survey_prompt("Q66", method="value", country="Ghana", persona=persona)
examples = fewshot_persona(answers)
fewshot = build_survey_prompt(
    "Q66", method="fewshot", country="Ghana", fewshot_responses=examples
)
country_only = build_survey_prompt("Q66", method="country", country="Ghana")
generic = build_survey_prompt("Q66", method="generic")
guidance = build_guidance_prompt("Q164", "Ghana", guidance="social_science")
generator = descriptor_generation_prompt("Q164", variant="composable")
```

For sociodemographic prompting, pass the persona text with `method="sociodemographic"`. Descriptor-generation messages combine the manuscript's instructions with the relevant survey question. Archive export requires the local preparation step above; template construction can also use caller-supplied answers.

## Example: collect ten responses and plot them

`collect_responses.py` submits exported messages to an OpenAI or compatible Chat Completions endpoint. It collects the main study's single-integer survey answers and token log-probabilities. Use a model that supports `logprobs`, `top_logprobs`, and `temperature=1`.

```bash
python -m pip install openai
python prompts.py --output exports/prompts.jsonl
# OPENAI_API_KEY must be set in the environment.
python collect_responses.py \
  --prompts exports/prompts.jsonl \
  --model gpt-5.4-nano \
  --reasoning-effort none \
  --limit 10 \
  --output results/gpt54_nano_10.parquet
python plot_sample.py \
  --input results/gpt54_nano_10.parquet \
  --output results/sample_distributions.png
```

After preparing the local data, run the commands above in the activated virtual environment. They collect one response for each of the first ten exported prompts using GPT-5.4 nano, then save one figure with ten question subplots to `results/sample_distributions.png`. Open that image to inspect the results. Omitting `--limit` collects every prompt in the input. Use `--limit 10` for the small example above. Models with reasoning controls can use `--reasoning-effort none` when supported, or a larger `--max-tokens` budget.

The Parquet output preserves the response-table schema, including question, country, persona, method, and source metadata. A companion `.raw.jsonl` file records the exact messages, request parameters, and API responses. Successful rows are saved after each request, and existing output files are never overwritten. To use custom messages, keep the exported record structure and update its `metadata` to describe the new condition; omit `prompt_id` for a new prompt.

```python
from analysis import load_responses, aggregate

responses = load_responses("results/gpt54_nano_10.parquet")
mean_distributions = aggregate(responses)
```

The example requests the top five alternative tokens at each position. The collector combines numeric token spellings for the same answer, and the analysis normalizes over the reported valid answer tokens. These are truncated token-based estimates, not guaranteed full categorical probabilities. Non-integer replies, refusals, truncated completions, or responses without token probabilities stop the example with an error. This example covers the main survey-response schema; guidance inputs use their own recorded schema.

Each subplot shows the mean normalized answer probabilities for that question, with its question text, answer labels, and number of collected responses. These are token-probability estimates, not frequencies of sampled answers. If the input contains multiple prompting conditions or models, they are averaged together within each question. Use a `.pdf` output filename for a PDF.

Run the collection checks without credentials or network access:

```bash
python test_collection.py
```

API fields follow the [official Chat Completions reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).

Guidance identifiers use the paper's names throughout the templates and response data: `social_science`, `population_prior`, `survey_context`, `first_person`, `predict`, `practical_judgment`, `minimal`, `life_shaped`, `decisive`, `identity_anchor`, `demographic_realism`, `lived_experience`, `cultural_norms`, `affective_gut`, and `contrastive`. Templates are grouped by context (`persona`, `country`, and `persona_no_country`).

Descriptor sources combine the paper's variant names (`composable`, `modular`, `no_hedge`) with the generator (`claude_sonnet_4_6`, `gpt_5_4`), for example `composable_claude_sonnet_4_6`.

## Package structure

| File | Purpose |
| --- | --- |
| `figures.py` | Plotting functions and command-line runner |
| `analysis.py` | Probability decoding, aggregation, errors, and calibration |
| `prompts.py` | Prompt construction and local archive export |
| `prepare_data.py` | Build survey-derived inputs from official EVS/WVS files |
| `demographics.py` | Reconstruct sociodemographic persona wording |
| `build_release.py` | Package public files, excluding local survey data |
| `test_reproduction.py` | Numerical invariants and archive integrity checks |
| `collect_responses.py` | Optional API collection example |
| `plot_sample.py` | Plot a collected sample as a grid of question distributions |
| `test_collection.py` | Collection checks with mocked API responses |
| `test_preparation.py` | Survey preparation and public archive checks |
| `data/responses.parquet` | Recorded token log-probabilities and experimental conditions |
| `data/human_counts.csv` | Locally generated country/question response counts |
| `data/guidance_responses.parquet` | Recorded guidance-experiment responses |
| `data/*selection.csv` | Historical sample identifiers, without answers |
| `data/prompt_conditions.csv` | Prompt conditions linked to saved LLM results |
| `data/prompt_index.parquet`, `data/prompt_text.json` | Locally generated prompt index and messages |
| `data/*.json` | Study definitions, question text, descriptors, and templates |

Response tables use release-local persona IDs. `respondent_selection.csv` maps them to official survey identifiers so users with the source data can reconstruct the study. General descriptor tables define text for each possible answer; individual combinations are generated locally. A project license has not been selected.

To build a shareable ZIP, run `python build_release.py`. It includes the six figure PDFs and an explicit list of public code and data files, regardless of which local outputs exist.
