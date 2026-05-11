"""
Build a semi-synthetic Criteo ITE dataset with potential outcomes.

The script reads the raw Criteo uplift CSV and generates:
- outcome_t0, outcome_t1 (potential outcomes)
- outcome_factual, outcome_counterfactual
- tau (individual treatment effect)

Generation follows the semi-synthetic surfaces used in
`criteo-ITE-Experiment.ipynb`.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = [f"f{i}" for i in range(12)]
REQUIRED_RAW = FEATURE_COLS + ["treatment"]


def repo_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _validate_and_extract(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    work = df[FEATURE_COLS + ["treatment"]].copy()
    for c in FEATURE_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce").astype("float64")
    work["treatment"] = pd.to_numeric(work["treatment"], errors="coerce").astype("Int64")
    work = work.dropna(subset=FEATURE_COLS + ["treatment"]).copy()
    work["treatment"] = work["treatment"].astype("int8")

    if not set(work["treatment"].unique()).issubset({0, 1}):
        raise ValueError(f"Treatment is not binary: {sorted(work['treatment'].unique())}")

    x = work[FEATURE_COLS].to_numpy(dtype=np.float64)
    t = work["treatment"].to_numpy(dtype=np.int8)
    return x, t


def _linear_surface(x: np.ndarray, rng: np.random.Generator, tau: float, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    beta_a = rng.choice([0, 1, 2, 3, 4], size=d, replace=True, p=[0.5, 0.2, 0.15, 0.1, 0.05]).astype(np.float64)
    mu0 = x @ beta_a
    mu1 = mu0 + tau
    y0 = rng.normal(loc=mu0, scale=noise_scale, size=n)
    y1 = rng.normal(loc=mu1, scale=noise_scale, size=n)
    return y0, y1


def _exp_surface(
    x: np.ndarray, t: np.ndarray, rng: np.random.Generator, tau: float, noise_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    beta_b = rng.choice([0.0, 0.1, 0.2, 0.3, 0.4], size=d, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    w = 0.5 * np.ones_like(x)
    mu0 = np.exp((x + w) @ beta_b)
    mu1 = x @ beta_b
    omega = np.mean(mu1[t == 1] - mu0[t == 1]) - tau
    mu1 = mu1 - omega
    y0 = rng.normal(loc=mu0, scale=noise_scale, size=n)
    y1 = rng.normal(loc=mu1, scale=noise_scale, size=n)
    return y0, y1


def _multi_exp_surface(
    x: np.ndarray,
    rng: np.random.Generator,
    ate_target: float,
    noise_scale: float,
    nb_centers: int,
    std_scaler: float,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, _ = x.shape
    stds = std_scaler * np.ones(nb_centers, dtype=np.float64)
    w0_init = rng.uniform(low=0.0, high=1.0, size=nb_centers)
    w1 = rng.uniform(low=0.0, high=1.0, size=nb_centers)
    centers = x[rng.choice(n, size=nb_centers, replace=False), :]

    def _mu(weights: np.ndarray, sqdist: bool) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        for i in range(nb_centers):
            dist = np.linalg.norm(centers[i, :] - x, axis=1)
            if sqdist:
                out += weights[i] * np.exp(-(dist**2) / (2.0 * (stds[i] ** 2)))
            else:
                out += weights[i] * np.exp(-dist / stds[i])
        return out

    mu1 = _mu(w1, sqdist=True)
    low, high = -8.0, 8.0
    mid = 0.5 * (low + high)
    mu0 = _mu(w0_init + mid, sqdist=False)
    ate = float(np.mean(mu1 - mu0))

    max_iter = 200
    it = 0
    while abs(ate - ate_target) > tol and it < max_iter:
        if ate > ate_target:
            low = mid
        else:
            high = mid
        mid = 0.5 * (low + high)
        mu0 = _mu(w0_init + mid, sqdist=False)
        ate = float(np.mean(mu1 - mu0))
        it += 1

    y0 = rng.normal(loc=mu0, scale=noise_scale, size=n)
    y1 = rng.normal(loc=mu1, scale=noise_scale, size=n)
    return y0, y1


def build_criteo_ite(
    raw_df: pd.DataFrame,
    synthetic_setup: str,
    ate_real: float,
    nb_centers: int,
    std_scaler: float,
    seed: int,
) -> pd.DataFrame:
    x, t = _validate_and_extract(raw_df)
    rng = np.random.default_rng(seed)
    noise_scale = ate_real / 4.0
    tol = ate_real / 100.0

    if synthetic_setup == "multi_exp":
        y0, y1 = _multi_exp_surface(
            x=x,
            rng=rng,
            ate_target=ate_real,
            noise_scale=noise_scale,
            nb_centers=nb_centers,
            std_scaler=std_scaler,
            tol=tol,
        )
    elif synthetic_setup == "linear":
        y0, y1 = _linear_surface(x=x, rng=rng, tau=ate_real, noise_scale=noise_scale)
    elif synthetic_setup == "exponential":
        y0, y1 = _exp_surface(x=x, t=t, rng=rng, tau=ate_real, noise_scale=noise_scale)
    else:
        raise ValueError("synthetic_setup must be one of: multi_exp, linear, exponential")

    factual = np.where(t == 1, y1, y0)
    counterfactual = np.where(t == 1, y0, y1)
    tau = y1 - y0

    out = pd.DataFrame(x, columns=FEATURE_COLS)
    out["treatment"] = t.astype(np.int8)
    out["outcome_t0"] = y0.astype(np.float64)
    out["outcome_t1"] = y1.astype(np.float64)
    out["outcome_factual"] = factual.astype(np.float64)
    out["outcome_counterfactual"] = counterfactual.astype(np.float64)
    out["tau"] = tau.astype(np.float64)
    return out


def main() -> None:
    default_data = repo_data_dir()
    parser = argparse.ArgumentParser(description="Create semi-synthetic Criteo ITE CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_data / "criteo-research-uplift-v2.1.csv",
        help="Raw Criteo file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data / "criteo_ite.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--synthetic_setup",
        type=str,
        choices=("multi_exp", "linear", "exponential"),
        default="multi_exp",
        help="Semi-synthetic surface type from the notebook.",
    )
    parser.add_argument("--ate_real", type=float, default=4.0, help="Target average treatment effect.")
    parser.add_argument("--nb_centers", type=int, default=5, help="Number of centers for multi_exp setup.")
    parser.add_argument("--std_scaler", type=float, default=1.0, help="Kernel std scale for multi_exp setup.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")


    raw_df = pd.read_csv(args.input)
    out_df = build_criteo_ite(
        raw_df=raw_df,
        synthetic_setup=args.synthetic_setup,
        ate_real=args.ate_real,
        nb_centers=args.nb_centers,
        std_scaler=args.std_scaler,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(
        f"Saved {args.output} shape={out_df.shape} "
        f"treated_rate={out_df['treatment'].mean():.4f} "
        f"tau_mean={out_df['tau'].mean():.4f}"
    )


if __name__ == "__main__":
    main()

