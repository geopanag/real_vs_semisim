"""
Clean Criteo uplift CSV: numeric f0..f11, binary treatment/outcomes, drop exposure.

Writes criteo_uplift_{outcome}.csv with columns f0..f11, treatment, and the chosen Y.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def repo_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


FEATURE_COLS = [f"f{i}" for i in range(12)]
REQUIRED_RAW = FEATURE_COLS + ["treatment", "conversion", "visit"]


def process_criteo_raw(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    work_df = df[FEATURE_COLS + ["treatment", "conversion", "visit"]].copy()

    for c in FEATURE_COLS:
        work_df[c] = pd.to_numeric(work_df[c], errors="coerce").astype("float32")

    work_df["treatment"] = pd.to_numeric(work_df["treatment"], errors="coerce").astype("Int64")
    work_df["conversion"] = pd.to_numeric(work_df["conversion"], errors="coerce").astype("Int64")
    work_df["visit"] = pd.to_numeric(work_df["visit"], errors="coerce").astype("Int64")

    work_df = work_df.dropna(subset=FEATURE_COLS + ["treatment", "conversion", "visit"]).copy()

    work_df["treatment"] = work_df["treatment"].astype("int8")
    work_df["conversion"] = work_df["conversion"].astype("int8")
    work_df["visit"] = work_df["visit"].astype("int8")

    if not set(work_df["treatment"].unique()).issubset({0, 1}):
        raise ValueError(f"Treatment is not binary: {sorted(work_df['treatment'].unique())}")

    for ycol in ("conversion", "visit"):
        if not set(work_df[ycol].unique()).issubset({0, 1}):
            raise ValueError(f"{ycol} is not binary: {sorted(work_df[ycol].unique())}")

    return work_df


def save_clean(work_df: pd.DataFrame, out_path: Path, outcome: str) -> None:
    if outcome not in ("conversion", "visit"):
        raise ValueError("outcome must be 'conversion' or 'visit'")
    clean_df = work_df[FEATURE_COLS + ["treatment", outcome]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False)


def main() -> None:
    default_data = repo_data_dir()
    parser = argparse.ArgumentParser(description="Preprocess Criteo uplift v2.1 CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_data / "criteo-uplift-v2.1.csv",
        help="Raw Criteo CSV (expects f0..f11, treatment, conversion, visit, exposure).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=default_data,
        help="Directory for cleaned outputs.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="criteo_uplift",
        help="Output basename prefix (files: {prefix}_conversion.csv, etc.).",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        choices=("conversion", "visit", "both"),
        default="conversion",
        help="Which outcome column to keep in the saved file (besides f* and treatment).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    work_df = process_criteo_raw(df)

    outcomes = ("conversion", "visit") if args.outcome == "both" else (args.outcome,)
    for oc in outcomes:
        out_path = args.out_dir / f"{args.prefix}_{oc}.csv"
        save_clean(work_df, out_path, oc)
        sub = work_df[FEATURE_COLS + ["treatment", oc]]
        print(f"Saved {out_path}  shape={sub.shape}  treated_rate={sub['treatment'].mean():.4f}  {oc}_rate={sub[oc].mean():.4f}")


if __name__ == "__main__":
    main()
