# Methods and reproducibility

The six figure commands combine packaged LLM responses with human references generated locally by `prepare_data.py` from official WVS Trends v4.1 and EVS ZA7503 v3.0.0 files. Figure layouts and label placement may differ slightly from the manuscript. The main heatmap's 572 rounded numerical annotations have been checked against the manuscript in their displayed order.

## Analysis conventions

- Valid answer-token log-probabilities are exponentiated and normalized. Ten-point responses use tokens 0–9, mapped to survey values 1–10.
- Model response distributions are averaged across personas before errors are calculated. Absolute expected-value errors are divided by the question's response range and averaged over questions within countries.
- The variance figure uses shared persona/question conditions across all 12 models. Variances are normalized by the maximum possible variance on the response scale.
- Calibration evaluates 31 temperatures from $10^{-1}$ to $10^2$. At temperatures other than 1, category probabilities receive additive smoothing of $10^{-3}$. Temperature 1 is an exact identity.
- Mean-preserving scaling uses an exponential tilt and bisection with expected-value tolerance $10^{-6}$.
- Temperature curves average responses within each model, method, question, and country. Sociodemographic responses are selected by method rather than descriptor-source label.
- The guidance figure uses nine open-model verification files, including E2B and Ministral 3B. For the two postmaterialism items, error is half the mean absolute difference between category probabilities. Other items use normalized expected-value error.
- Human distributions use unweighted counts within the selected sample.
- The Ghana Q66 example is calculated directly from the packaged response probabilities and calibration output.

## Prompt conventions

The preparation script reconstructs participant-specific messages from the official survey records. Garbled apostrophes and em dashes in the historical prompts have been corrected to Unicode punctuation; recorded LLM outputs have not been regenerated. The archive exporter then expands the local prompt index into system and user messages. Main-study user messages are checked against the template builder, including whitespace and punctuation. Ten-point question text uses shifted numeric anchors while the printed `Scale:` labels retain their recorded values.

The package includes six descriptor tables and 15 guidance variants. Descriptor-generation messages combine the instructions in the manuscript with the question text. Sociodemographic prompting takes persona text supplied by the caller.

## Validation

Run `python test_reproduction.py` to check token decoding, probability normalization, preservation of expected values, pair-wise holdout selection, response/prompt references, exact archived-message expansion, prompt construction, and complete-case selection.

## Local survey preparation

The public sample-selection tables contain dataset and respondent identifiers, not answers. They preserve the historical selection of 4,860 human-reference respondents and 97 personas. All descriptor, few-shot, and sociodemographic persona messages are assembled locally. The human-reference counts are computed from the selected records without weights.

Generated human counts, prompt text, and prompt indexes are excluded from Git and the public ZIP.

The local rebuild was compared with the original release inputs: all 13,930 prompt message pairs matched after correcting garbled punctuation, and all human counts matched exactly. Run `python -m unittest test_reproduction test_collection test_preparation` after preparation to check the analysis, collection example, reconstruction helpers, and archive exclusions.
