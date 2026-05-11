"""
Build Hillstrom men vs control and women vs control tables with encoded features.

Reads raw hillstrom.csv, validates segment labels, adds treatment, encodes
history_segment (ordinal), zip_code and channel (one-hot), then writes
 hillstrom_men.csv and hillstrom_women.csv under data/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from sklift.datasets import fetch_hillstrom
import pandas as pd


def repo_data_dir() -> Path:
    """.../causalml/data (this file lives in code/cate_estimation/)."""
    return Path(__file__).resolve().parents[2] / "data"


def history_segment_to_ordinal(s: pd.Series) -> pd.Series:
    """Map '3) $200 - $350' -> 3, etc."""
    out = s.astype(str).str.extract(r"^(\d+)\)", expand=False)
    return pd.to_numeric(out, errors="coerce").astype("Int64")


def encode_hillstrom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal encode history_segment; one-hot zip_code and channel.
    Drops raw history_segment after adding history_segment_ord.
    """
    out = df.copy()
    out["history_segment_ord"] = history_segment_to_ordinal(out["history_segment"])
    out = pd.get_dummies(
        out,
        columns=["zip_code", "channel"],
        prefix=["zip", "ch"],
        drop_first=False,
        dtype=int,
    )
    return out.drop(columns=["history_segment"])


def load_and_validate(path: Path, segment_col: str = "segment") -> pd.DataFrame:
    df = pd.read_csv(path)
    if segment_col not in df.columns:
        raise ValueError(
            f"Column '{segment_col}' not found. Available columns: {list(df.columns)}"
        )
    valid_segments = {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}
    found_segments = set(df[segment_col].dropna().unique())
    if not found_segments.issubset(valid_segments):
        raise ValueError(
            f"Unexpected segment values: {found_segments}. "
            f"Expected subset of {valid_segments}"
        )
    return df


def preprocess(
    input_csv: Path,
    output_mens: Path,
    output_womens: Path,
    segment_col: str = "segment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_and_validate(input_csv, segment_col=segment_col)

    df_mens = df[df[segment_col].isin(["No E-Mail", "Mens E-Mail"])].copy()
    df_mens["treatment"] = (df_mens[segment_col] == "Mens E-Mail").astype(int)

    df_womens = df[df[segment_col].isin(["No E-Mail", "Womens E-Mail"])].copy()
    df_womens["treatment"] = (df_womens[segment_col] == "Womens E-Mail").astype(int)

    df_mens_enc = encode_hillstrom_features(df_mens)
    df_womens_enc = encode_hillstrom_features(df_womens)

    output_mens.parent.mkdir(parents=True, exist_ok=True)
    df_mens_enc.to_csv(output_mens, index=False)
    df_womens_enc.to_csv(output_womens, index=False)

    return df_mens_enc, df_womens_enc


def main() -> None:
    default_data = repo_data_dir()
    parser = argparse.ArgumentParser(description="Preprocess Hillstrom CSV into men/women arms.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_data / "hillstrom.csv",
        help="Raw Hillstrom CSV path.",
    )
    parser.add_argument(
        "--output_mens",
        type=Path,
        default=default_data / "hillstrom_men.csv",
        help="Output path for men vs control (encoded).",
    )
    parser.add_argument(
        "--output_womens",
        type=Path,
        default=default_data / "hillstrom_women.csv",
        help="Output path for women vs control (encoded).",
    )
    parser.add_argument("--segment_col", type=str, default="segment")
    args = parser.parse_args()

    df_m, df_w = preprocess(
        args.input,
        args.output_mens,
        args.output_womens,
        segment_col=args.segment_col,
    )


    
    dataset = fetch_hillstrom(target_col='visit')
    print(dataset.head())
    print(dataset.columns)
    print(dataset.shape)
    print(dataset.info())
    print(dataset.describe())
    print(dataset.head())
    print(dataset.tail())
    print(dataset.sample(10))
    print(dataset.sample(10))
    print("Saved:", args.output_mens, df_m.shape)
    print("Treatment counts (men):")
    print(df_m["treatment"].value_counts(dropna=False).sort_index())
    print()
    print("Saved:", args.output_womens, df_w.shape)
    print("Treatment counts (women):")
    print(df_w["treatment"].value_counts(dropna=False).sort_index())
    print()
    print("Encoded columns (prefix sample):", [c for c in df_m.columns if c.startswith(("zip_", "ch_", "history_segment"))])


if __name__ == "__main__":
    main()
