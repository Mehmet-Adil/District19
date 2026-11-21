"""Create separate week-6 sensitivity submissions from a chosen forecast CSV."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "submission_full_lightgbm_young_model.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs" / "week6_sensitivity"
FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def main() -> None:
    forecast = pd.read_csv(SOURCE)
    if "article_id" not in forecast or "w6" not in forecast:
        raise ValueError("Expected article_id and w6 columns in source submission")
    if forecast["article_id"].duplicated().any():
        raise ValueError("article_id must be unique")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for factor in FACTORS:
        variant = forecast.copy()
        variant["w6"] = variant["w6"] * factor
        target = OUTPUT / f"week6_x{factor:g}.csv"
        variant.to_csv(target, index=False)
        print(f"{target}: w6 mean {forecast.w6.mean():.3f} -> {variant.w6.mean():.3f}")


if __name__ == "__main__":
    main()
