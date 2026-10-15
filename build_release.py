"""Build the public release without local survey records or generated prompts."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = [
    ".gitignore", ".gitattributes", "README.md", "REPRODUCIBILITY.md", "requirements.txt", "requirements-data.txt",
    "analysis.py", "figures.py", "prompts.py",
    "demographics.py", "prepare_data.py", "collect_responses.py", "plot_sample.py",
    "build_release.py", "test_reproduction.py", "test_collection.py", "test_preparation.py",
    "data/responses.parquet", "data/guidance_responses.parquet",
    "data/descriptors.json", "data/guidance_templates.json",
    "data/questions.json", "data/inverted_questions.json", "data/study.json",
    "data/survey_columns.json", "data/respondent_selection.csv", "data/prompt_conditions.csv",
    "figures/heatmap_mae.pdf", "figures/peak_distributions.pdf",
    "figures/guidance_comparison.pdf", "figures/error_comparison.pdf",
    "figures/temperature_all_curves.pdf", "figures/q66_histograms_ghana.pdf",
]


def build_release(output):
    """Use an explicit allowlist so local outputs cannot enter the archive."""
    missing = [name for name in PUBLIC_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing public release files: {', '.join(missing)}")
    output = Path(output).resolve()
    if output in {(ROOT / name).resolve() for name in PUBLIC_FILES}:
        raise ValueError("Archive output must not overwrite a release source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in PUBLIC_FILES:
            archive.write(ROOT / name, f"public_release/{name}")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Release ZIP failed its integrity check")
    temporary.replace(output)
    print(f"Packaged {len(PUBLIC_FILES)} public files in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "personas_public_release.zip")
    build_release(parser.parse_args().output)


if __name__ == "__main__":
    main()
