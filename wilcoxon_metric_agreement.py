from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent

# Default column mapping; override via CLI if files differ.
DEFAULT_COLUMNS = {
    "dataset": "dataset",
    "benchmark_instance": "benchmark_instance",
    "seed": "seed",
    "method": "method",
    "metric": "metric",
    "value": "value",
}

# Canonical metric aliases.
METRIC_ALIASES = {
    "uplift20": "uplift20",
    "uplift@20": "uplift20",
    "uplift @20": "uplift20",
    "uplift_at_20": "uplift20",
    "qini": "qini",
    "rate": "rate",
    "factual": "factual",
    "factual_loss": "factual",
    "delta ate": "delta_ate",
    "d ate": "delta_ate",
    "delta_ate": "delta_ate",
    "deltaate": "delta_ate",
    "delta ate sim": "delta_ate_sim",
    "d ate sim": "delta_ate_sim",
    "delta_ate_sim": "delta_ate_sim",
    "deltaatesim": "delta_ate_sim",
    "sqrt_pehe": "sqrt_pehe",
    "sqrt pehe": "sqrt_pehe",
    "sqrt(pehe)": "sqrt_pehe",
    "policy": "policy",
}

# Higher-is-better metrics (direction handling skipped with --input_is_rank).
HIGHER_IS_BETTER = {"uplift20", "qini", "rate", "policy"}

OBS_METRICS_BASE = {"uplift20", "qini", "rate", "delta_ate", "factual"}
CF_METRICS = {"sqrt_pehe", "policy", "delta_ate_sim"}

TAG_RE = re.compile(
    r"^(?P<dataset>ihdp|acic2016|twins)_"
    r"(?P<model_name>[^_]+)_"
    r"k(?P<training_split>\d+)_"
    r"train(?P<train_token>[^_]+)_"
    r"(?P<metalearn>[^_]+)_"
    r"seed(?P<seed>\d+)"
    r"(?:_exp(?P<exp>\d+))?$"
)

WIDE_METRIC_COLUMN_MAP = {
    "up20": "uplift20",
    "qini_score": "qini",
    "rate_autoc": "rate",
    "factual_loss": "factual",
    "abs_error_ATE": "delta_ate_sim",
    "sqrt_PEHE": "sqrt_pehe",
    "policy_value_from_tau": "policy",
    # Often used as an approximation for simulation-specific ATE mismatch.
    "uplift_diff": "delta_ate",
}


def _clean_metric_name(metric: str) -> str:
    text = str(metric).strip().lower()
    text = text.replace("\\", " ")
    text = text.replace("$", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("mathrm", " ")
    text = text.replace("delta", "delta")
    text = text.replace("Δ", "delta")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("(sim)", " sim")
    text = text.replace("simulated", "sim")
    text = text.replace("@", "@")
    return text


def normalize_metric(metric: str) -> str | None:
    raw = _clean_metric_name(metric)
    raw_key = raw.replace(" ", "")
    if raw in METRIC_ALIASES:
        return METRIC_ALIASES[raw]
    if raw_key in METRIC_ALIASES:
        return METRIC_ALIASES[raw_key]
    if raw == "delta ate sim":
        return "delta_ate_sim"
    if raw == "delta ate":
        return "delta_ate"
    if raw in {"sqrt pehe", "sqrt(pehe)"}:
        return "sqrt_pehe"
    return None


def _method_label(metalearn: str, model_name: str) -> str:
    if metalearn.lower() in {"cfr", "tarnet", "dragon", "causalforest"}:
        return metalearn
    return f"{metalearn} ({model_name})"


def _rows_from_wide_results_file(csv_path: Path, force_dataset: str | None = None) -> list[dict[str, object]]:
    name = csv_path.name
    if not (name.startswith("results_") and name.endswith(".csv")):
        return []
    tag = name[len("results_") : -len(".csv")]
    if tag.endswith("_optimized"):
        tag = tag[: -len("_optimized")]

    m = TAG_RE.match(tag)
    if m is None:
        return []
    gd = m.groupdict()
    dataset = (force_dataset or gd["dataset"]).lower()
    if dataset not in {"ihdp", "acic2016", "twins"}:
        return []

    seed = int(gd["seed"])
    bench = gd["exp"] if gd["exp"] is not None else "1"
    method = _method_label(gd["metalearn"], gd["model_name"])

    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    row = df.iloc[0]

    out: list[dict[str, object]] = []
    for src_col, canonical_metric in WIDE_METRIC_COLUMN_MAP.items():
        if src_col not in df.columns:
            continue
        val = pd.to_numeric(row[src_col], errors="coerce")
        if pd.isna(val):
            continue
        out.append(
            {
                "dataset": dataset,
                "benchmark_instance": str(bench),
                "seed": seed,
                "method": method,
                "metric": canonical_metric,
                "value": float(val),
            }
        )
    return out


def _load_default_benchmark_results(results_sim_root: Path, twins_results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for dataset in ("ihdp", "acic2016"):
        ds_dir = results_sim_root / dataset
        if not ds_dir.is_dir():
            print(f"[WARN] Missing dataset folder under results_sim: {ds_dir}")
            continue
        for p in ds_dir.iterdir():
            if not p.is_file() or p.suffix.lower() != ".csv":
                continue
            rows.extend(_rows_from_wide_results_file(p, force_dataset=dataset))

    if twins_results_root.is_dir():
        for p in twins_results_root.rglob("results_*.csv"):
            rows.extend(_rows_from_wide_results_file(p, force_dataset="twins"))
    else:
        print(f"[WARN] twins_results_root not found: {twins_results_root}")

    if not rows:
        raise ValueError("No rows loaded from default benchmark paths.")
    return pd.DataFrame(rows)


def method_to_strategy(method: str) -> str:
    m = str(method).strip()
    upper = m.upper()
    compact = re.sub(r"\s+", "", upper)
    compact = compact.replace("-", "")
    if "CAUSALFOREST" in compact:
        return "Causal Forest"
    if "DRAGON" in compact:
        return "DragonNet"
    if "TARNET" in compact:
        return "TARNet"
    if compact.startswith("CFR") or "CFR" in compact:
        return "CFR"
    if compact.startswith("DR"):
        return "DR"
    if compact.startswith("R"):
        return "R"
    if compact.startswith("X"):
        return "X"
    if compact.startswith("T"):
        return "T"
    if compact.startswith("S"):
        return "S"
    return m


def pair_type(metric_i: str, metric_j: str, obs_set: set[str], cf_set: set[str]) -> str:
    in_obs_i, in_obs_j = metric_i in obs_set, metric_j in obs_set
    in_cf_i, in_cf_j = metric_i in cf_set, metric_j in cf_set
    if in_obs_i and in_obs_j:
        return "obs_obs"
    if (in_obs_i and in_cf_j) or (in_cf_i and in_obs_j):
        return "obs_cf"
    if in_cf_i and in_cf_j:
        return "cf_cf"
    return "other"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wilcoxon signed-rank test for agreement between observable and "
            "counterfactual metric rankings on semi-simulated CATE benchmarks."
        )
    )
    parser.add_argument(
        "--input_files",
        type=Path,
        nargs="+",
        default=None,
        help="Optional one or more CSV files in long format. If omitted, auto-load from benchmark roots.",
    )
    parser.add_argument(
        "--results_sim_root",
        type=Path,
        default=Path("./results_sim"),
        help="Root for ihdp/acic2016 simulated results.",
    )
    parser.add_argument(
        "--twins_results_root",
        type=Path,
        default=Path("./results"),
        help="Root for twins results files.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./analysis_outputs"))
    parser.add_argument("--prefix", type=str, default="metric_agreement")

    parser.add_argument("--dataset_col", type=str, default=DEFAULT_COLUMNS["dataset"])
    parser.add_argument(
        "--benchmark_instance_col",
        type=str,
        default=DEFAULT_COLUMNS["benchmark_instance"],
    )
    parser.add_argument("--seed_col", type=str, default=DEFAULT_COLUMNS["seed"])
    parser.add_argument("--method_col", type=str, default=DEFAULT_COLUMNS["method"])
    parser.add_argument("--metric_col", type=str, default=DEFAULT_COLUMNS["metric"])
    parser.add_argument("--value_col", type=str, default=DEFAULT_COLUMNS["value"])

    parser.add_argument("--input_is_rank", action="store_true")
    parser.add_argument("--strategy_level", action="store_true")
    parser.add_argument("--ranking_only_obs", action="store_true")
    parser.add_argument("--exclude_factual", action="store_true")
    parser.add_argument("--min_common_methods", type=int, default=5)
    parser.add_argument(
        "--zero_method",
        type=str,
        default="wilcox",
        choices=["wilcox", "pratt", "zsplit"],
        help="scipy.stats.wilcoxon zero_method",
    )

    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--plots_dir", type=Path, default=None)
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Print only Wilcoxon summary to stdout; still writes CSV/JSON files.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = args.plots_dir.expanduser().resolve() if args.plots_dir else (output_dir / "plots")
    if args.make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    if args.input_files:
        frames: list[pd.DataFrame] = []
        for f in args.input_files:
            fp = f.expanduser().resolve()
            if not fp.is_file():
                raise FileNotFoundError(f"Missing input file: {fp}")
            frames.append(pd.read_csv(fp))
        df = pd.concat(frames, ignore_index=True)

        required_cols = [
            args.dataset_col,
            args.benchmark_instance_col,
            args.seed_col,
            args.method_col,
            args.metric_col,
            args.value_col,
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        work = df[required_cols].rename(
            columns={
                args.dataset_col: "dataset",
                args.benchmark_instance_col: "benchmark_instance",
                args.seed_col: "seed",
                args.method_col: "method",
                args.metric_col: "metric",
                args.value_col: "value",
            }
        )
    else:
        work = _load_default_benchmark_results(
            results_sim_root=args.results_sim_root.expanduser().resolve(),
            twins_results_root=args.twins_results_root.expanduser().resolve(),
        )
        print("Loaded input automatically from results_sim/results roots.")
    work["dataset"] = work["dataset"].astype(str).str.strip().str.lower()
    work = work[work["dataset"].isin({"ihdp", "acic2016", "twins"})].copy()
    work["metric_original"] = work["metric"].astype(str)
    work["metric"] = work["metric_original"].map(normalize_metric)
    unrecognized = sorted(work.loc[work["metric"].isna(), "metric_original"].dropna().unique().tolist())
    if unrecognized:
        print(f"[WARN] Unrecognized metrics (dropped): {unrecognized}")
    work = work.dropna(subset=["metric"]).copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value"])

    recognized_metrics = sorted(work["metric"].unique().tolist())
    print(f"Recognized metrics: {recognized_metrics}")

    obs_metrics = set(OBS_METRICS_BASE)
    if args.ranking_only_obs:
        obs_metrics = {"uplift20", "qini", "rate"}
    if args.exclude_factual:
        obs_metrics.discard("factual")

    print(f"Observable metrics set: {sorted(obs_metrics)}")
    print(f"Counterfactual metrics set: {sorted(CF_METRICS)}")
    print(f"Input interpreted as {'ranks' if args.input_is_rank else 'raw metric values'}")

    grouped = (
        work.groupby(["dataset", "benchmark_instance", "method", "metric"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "value_mean"})
    )

    grouped["rank_value"] = grouped["value_mean"]
    if not args.input_is_rank:
        higher_mask = grouped["metric"].isin(HIGHER_IS_BETTER)
        grouped.loc[higher_mask, "rank_value"] = -grouped.loc[higher_mask, "value_mean"]
        grouped.loc[~higher_mask, "rank_value"] = grouped.loc[~higher_mask, "value_mean"]

    grouped["rank"] = grouped.groupby(["dataset", "benchmark_instance", "metric"])["rank_value"].rank(
        method="average",
        ascending=True,
    )

    if args.strategy_level:
        grouped["entity"] = grouped["method"].map(method_to_strategy)
        entity_rank = (
            grouped.groupby(["dataset", "benchmark_instance", "metric", "entity"], as_index=False)["rank"]
            .mean()
            .rename(columns={"rank": "entity_rank"})
        )
        entity_col = "entity"
        rank_col = "entity_rank"
    else:
        grouped["entity"] = grouped["method"].astype(str)
        entity_rank = grouped[["dataset", "benchmark_instance", "metric", "entity", "rank"]].rename(
            columns={"rank": "entity_rank"}
        )
        entity_col = "entity"
        rank_col = "entity_rank"

    pair_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []

    for (dataset, bench), bench_df in entity_rank.groupby(["dataset", "benchmark_instance"], sort=True):
        metric_to_entities: dict[str, pd.DataFrame] = {}
        n_entities_per_metric: dict[str, int] = {}
        for metric, mdf in bench_df.groupby("metric", sort=True):
            tmp = mdf[[entity_col, rank_col]].dropna().drop_duplicates(subset=[entity_col], keep="last")
            metric_to_entities[metric] = tmp
            n_entities_per_metric[metric] = int(tmp[entity_col].nunique())

        metrics_present = sorted(metric_to_entities.keys())
        if len(metrics_present) < 2:
            print(f"[WARN] skip {dataset}/{bench}: fewer than 2 metrics present.")
            continue

        for metric_i, metric_j in combinations(metrics_present, 2):
            left = metric_to_entities[metric_i]
            right = metric_to_entities[metric_j]
            merged = left.merge(right, on=entity_col, suffixes=("_i", "_j"), how="inner")
            n_common = int(len(merged))
            if n_common < args.min_common_methods:
                continue
            tau, pval = kendalltau(merged[f"{rank_col}_i"], merged[f"{rank_col}_j"])
            ptype = pair_type(metric_i, metric_j, obs_metrics, CF_METRICS)
            pair_rows.append(
                {
                    "dataset": dataset,
                    "benchmark_instance": bench,
                    "metric_i": metric_i,
                    "metric_j": metric_j,
                    "metric_pair_type": ptype,
                    "kendall_tau": float(tau) if tau is not None else np.nan,
                    "p_value": float(pval) if pval is not None else np.nan,
                    "n_common_methods": n_common,
                }
            )

        bench_pairs = [r for r in pair_rows if r["dataset"] == dataset and r["benchmark_instance"] == bench]
        obs_obs_vals = [float(r["kendall_tau"]) for r in bench_pairs if r["metric_pair_type"] == "obs_obs" and pd.notna(r["kendall_tau"])]
        obs_cf_vals = [float(r["kendall_tau"]) for r in bench_pairs if r["metric_pair_type"] == "obs_cf" and pd.notna(r["kendall_tau"])]
        if not obs_obs_vals or not obs_cf_vals:
            print(
                f"[WARN] skip {dataset}/{bench}: insufficient obs_obs or obs_cf "
                f"pairs after min_common_methods={args.min_common_methods}."
            )
            continue

        tau_obs_obs = float(np.mean(obs_obs_vals))
        tau_obs_cf = float(np.mean(obs_cf_vals))
        d_val = tau_obs_obs - tau_obs_cf
        n_methods_average = float(np.mean(list(n_entities_per_metric.values()))) if n_entities_per_metric else np.nan
        instance_rows.append(
            {
                "dataset": dataset,
                "benchmark_instance": bench,
                "n_methods_average": n_methods_average,
                "tau_obs_obs": tau_obs_obs,
                "tau_obs_cf": tau_obs_cf,
                "D": d_val,
                "n_obs_obs_pairs": len(obs_obs_vals),
                "n_obs_cf_pairs": len(obs_cf_vals),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    inst_df = pd.DataFrame(instance_rows)

    instances_csv = output_dir / f"{args.prefix}_instances.csv"
    pairwise_csv = output_dir / f"{args.prefix}_pairwise_kendall.csv"
    summary_json = output_dir / f"{args.prefix}_summary.json"

    if not pair_df.empty:
        pair_df = pair_df.sort_values(["dataset", "benchmark_instance", "metric_i", "metric_j"]).reset_index(drop=True)
    pair_df.to_csv(pairwise_csv, index=False)

    if not inst_df.empty:
        inst_df = inst_df.sort_values(["dataset", "benchmark_instance"]).reset_index(drop=True)
    inst_df.to_csv(instances_csv, index=False)

    d_values = inst_df["D"].dropna().to_numpy(dtype=float) if "D" in inst_df.columns else np.array([])
    n_pos = int(np.sum(d_values > 0))
    n_neg = int(np.sum(d_values < 0))
    n_zero = int(np.sum(d_values == 0))

    wilcox_stat = None
    wilcox_p = None
    wilcox_error = None
    if len(d_values) > 0:
        try:
            wil = wilcoxon(d_values, alternative="greater", zero_method=args.zero_method)
            wilcox_stat = float(wil.statistic)
            wilcox_p = float(wil.pvalue)
        except ValueError as exc:
            wilcox_error = str(exc)

    by_dataset = {}
    if not inst_df.empty:
        for ds, ds_df in inst_df.groupby("dataset", sort=True):
            vals = ds_df["D"].dropna().to_numpy(dtype=float)
            by_dataset[ds] = {
                "n_instances": int(len(vals)),
                "mean_D": float(np.mean(vals)) if len(vals) else None,
                "median_D": float(np.median(vals)) if len(vals) else None,
            }

    summary = {
        "n_benchmark_instances": int(len(d_values)),
        "mean_D": float(np.mean(d_values)) if len(d_values) else None,
        "median_D": float(np.median(d_values)) if len(d_values) else None,
        "std_D": float(np.std(d_values, ddof=1)) if len(d_values) > 1 else None,
        "wilcoxon_zero_method": args.zero_method,
        "wilcoxon_statistic": wilcox_stat,
        "wilcoxon_one_sided_pvalue": wilcox_p,
        "wilcoxon_error": wilcox_error,
        "n_positive_D": n_pos,
        "n_negative_D": n_neg,
        "n_zero_D": n_zero,
        "per_dataset": by_dataset,
        "options": {
            "input_is_rank": bool(args.input_is_rank),
            "strategy_level": bool(args.strategy_level),
            "ranking_only_obs": bool(args.ranking_only_obs),
            "exclude_factual": bool(args.exclude_factual),
            "min_common_methods": int(args.min_common_methods),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.make_plots and not inst_df.empty:
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415

            plt.figure(figsize=(6, 4))
            plt.hist(inst_df["D"].dropna(), bins=20)
            plt.xlabel("D = tau_obs_obs - tau_obs_cf")
            plt.ylabel("Count")
            plt.title("Distribution of D across benchmark instances")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{args.prefix}_hist_D.png", dpi=150)
            plt.close()

            plt.figure(figsize=(6, 4))
            inst_df.boxplot(column="D", by="dataset")
            plt.suptitle("")
            plt.title("D by dataset")
            plt.ylabel("D")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{args.prefix}_boxplot_D_by_dataset.png", dpi=150)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.scatter(inst_df["tau_obs_obs"], inst_df["tau_obs_cf"], alpha=0.7)
            lo = float(np.nanmin([inst_df["tau_obs_obs"].min(), inst_df["tau_obs_cf"].min()]))
            hi = float(np.nanmax([inst_df["tau_obs_obs"].max(), inst_df["tau_obs_cf"].max()]))
            plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel("tau_obs_obs")
            plt.ylabel("tau_obs_cf")
            plt.title("Per-instance agreement: obs_obs vs obs_cf")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{args.prefix}_scatter_obsobs_vs_obscf.png", dpi=150)
            plt.close()
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Plotting failed/skipped: {exc}")

    if args.summary_only:
        print("Wilcoxon Signed-Rank Test (H1: median(D) > 0)")
        print(f"n_benchmark_instances={summary['n_benchmark_instances']}")
        print(f"mean_D={summary['mean_D']}")
        print(f"median_D={summary['median_D']}")
        print(f"wilcoxon_statistic={summary['wilcoxon_statistic']}")
        print(f"wilcoxon_one_sided_pvalue={summary['wilcoxon_one_sided_pvalue']}")
        print(
            f"counts: positive={summary['n_positive_D']} "
            f"negative={summary['n_negative_D']} zero={summary['n_zero_D']}"
        )
    else:
        print(f"Wrote: {instances_csv}")
        print(f"Wrote: {pairwise_csv}")
        print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
