import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold


from utils import (
    set_seed,
    evaluate,
    ensure_retailhero_preprocessed,
    load_hillstrom_csv,
    ensure_hillstrom_preprocessed,
    ensure_criteo_preprocessed,
    ensure_criteo_ite_preprocessed,
    load_criteo_csv,
    load_criteo_ite_csv,
    load_nhefs,
    load_acic2016,
    load_ihdp,
    load_twins,
)

from causal_benchmarks import (
    results_filename_suffix,
    run_causal_forest,
    run_cfr,
    run_dr_learner,
    run_dragonnet,
    run_r_learner,
    run_s_learner,
    run_t_learner,
    run_x_learner,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_PREDICTIONS_DIR = "./predictions"
IHDP_N_SIMULATIONS = 100
ACIC2016_N_SIMULATIONS = 10


def experiment_basename(
    dataset: str,
    model_name: str,
    training_split: int,
    train_frac_token: str,
    metalearn: str,
    seed: int,
) -> str:
    """Same tag string as results CSV basename (without ``results_`` prefix)."""
    return (
        f"{dataset}_{model_name}_k{training_split}_train{train_frac_token}_{metalearn}_"
        f"seed{seed}"
    )

def frac_experiment(
    *,
    frac: float,
    train_pool_indices: np.ndarray,
    test_indices: np.ndarray,
    fold_idx: int,
    dataset: str,
    model_name: str,
    training_split: int,
    metalearn: str,
    seed: int,
    validation_fraction_percent: float,
    xu: torch.Tensor,
    xp: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    basename_suffix: str,
    save_predictions: bool,
    dataset_predictions_root: str,
    outcome_task: int,
    po_test: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    sampled_train_pool = subsample_train_pool(train_pool_indices, frac, fold_idx=fold_idx, seed=seed)
    # Random train/val split on the pool (KFold train indices are ordered; frac=1.0 was purely prefix-based).
    frac_code = int(frac * 1000)
    split_rng = np.random.default_rng(seed + fold_idx * 1000 + frac_code + 17_000)
    shuffled_pool = split_rng.permutation(sampled_train_pool)
    val_size = int(len(shuffled_pool) * validation_fraction_percent)
    val_indices = shuffled_pool[:val_size]
    train_nodes = shuffled_pool[val_size:]

    X = xu.cpu().detach().numpy()

    train_indices = np.hstack([train_nodes, val_indices])
    treatment_np = treatment.clone().unsqueeze(1).cpu().detach().numpy()
    y = outcome.cpu().detach().numpy()

    pred_c = None
    pred_t = None
    uplift = None
    if metalearn == "S":
        pred_c, pred_t = run_s_learner(model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task)
    elif metalearn == "T":
        pred_c, pred_t = run_t_learner(model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task)
    elif metalearn == "X":
        pred_c, pred_t, uplift = run_x_learner(
            model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task
        )
    elif metalearn == "R":
        pred_c, pred_t, uplift = run_r_learner(
            model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task
        )
    elif metalearn == "DR":
        pred_c, pred_t, uplift = run_dr_learner(
            model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task
        )
    elif metalearn == "CFR":
        pred_c, pred_t, uplift = run_cfr(
            model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task
        )
    elif metalearn == "TARnet":
        pred_c, pred_t, uplift = run_cfr(
            model_name,
            X,
            y,
            treatment_np,
            train_indices,
            test_indices,
            task=outcome_task,
            alpha=0.0,
        )
    elif metalearn == "Dragon":
        pred_c, pred_t, uplift = run_dragonnet(
            model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task
        )
    elif metalearn == "CausalForest":
        pred_c, pred_t, uplift = run_causal_forest(
            model_name, X, y, treatment_np, train_indices, test_indices, task=outcome_task
        )

    if pred_t is None or pred_c is None:
        raise ValueError(f"Unknown metalearn='{metalearn}' (no predictions computed)")

    result_row = evaluate(
        treatment.detach().cpu(),
        outcome.detach().cpu(),
        pred_t,
        pred_c,
        test_indices,
        X=X,
        train_indices=train_indices,
        po_test=po_test,
        uplift=uplift,
        has_individual_outcomes=(metalearn != "CausalForest"),
    )

    if save_predictions:
        train_frac_token = {1.0: "1"}[frac]
        tag = experiment_basename(
            dataset,
            model_name,
            training_split,
            train_frac_token,
            metalearn,
            seed,
        ) + basename_suffix
        save_fold_predictions(
            dataset_predictions_root,
            tag,
            fold_idx,
            train_frac_token,
            test_indices,
            treatment.detach().cpu().numpy()[test_indices],
            outcome.detach().cpu().numpy()[test_indices],
            outcome_task,
            pred_c,
            pred_t,
            uplift=uplift,
            metadata={
                "dataset": dataset,
                "model_name": model_name,
                "training_split": training_split,
                "train_fraction_token": train_frac_token,
                "metalearn": metalearn,
                "seed": seed,
                "validation_fraction_percent": validation_fraction_percent,
                "outcome_task": outcome_task,
                "basename_suffix": basename_suffix or None,
            },
        )

    return result_row



def run_experiment(
    *,
    dataset: str,
    xu: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    xp: torch.Tensor,
    model_name: str,
    training_split: int,
    metalearn: str,
    seed: int,
    validation_fraction_percent: float,
    path_to_results: str,
    path_to_predictions: str,
    device: torch.device,
    basename_suffix: str = "",
    write_results_csv: bool = True,
    save_predictions: bool = True,
    po_full: Optional[np.ndarray] = None,
) -> Optional[Dict[float, pd.Series]]:
    """
    K-fold CV on ``xu``, train-fraction sub-sample of the training pool per fold
    (full pool only: ``frac == 1.0``), meta-learners, ``evaluate``, optional
    ``.npz`` predictions, and one CSV per fraction.

    Use ``basename_suffix`` (e.g. ``_exp042``) when repeating over simulated draws.

    If ``write_results_csv`` is False, returns a mapping ``frac -> Series`` of
    column means over K folds (for aggregating many simulations). Otherwise returns
    ``None`` after writing CSVs.
    """
    treatment = treatment.reshape(-1)
    outcome = outcome.reshape(-1)

    outcome_task = infer_outcome_task(outcome.detach().cpu().numpy())
    print(
        f" outcome_task={outcome_task} "
        f"(0=binary {{0,1}} / classifiers, 1=regression)"
        f"{(' [' + basename_suffix.lstrip('_') + ']') if basename_suffix else ''}"
    )
    dataset_predictions_root = str(Path(path_to_predictions) / dataset)
    print(" predictions dir: <path_to_predictions>/<dataset>/<experiment_tag>/foldXX_train<tok>.npz")

    train_fracs = [1.0]
    train_frac_token = {1.0: "1"}

    print(" k folding...")
    kf = KFold(n_splits=training_split, shuffle=True, random_state=seed)
    fold_idx = 0
    results_by_frac: Dict[float, list] = {frac: [] for frac in train_fracs}
    for train_pool_indices, test_indices in kf.split(xu):
        fold_idx += 1
        torch.cuda.empty_cache()
        po_test = None
        if po_full is not None:
            po_test = np.asarray(po_full, dtype=np.float64)[test_indices]
        for frac in train_fracs:
            result_row = frac_experiment(
                frac=frac,
                train_pool_indices=train_pool_indices,
                test_indices=test_indices,
                fold_idx=fold_idx,
                dataset=dataset,
                model_name=model_name,
                training_split=training_split,
                metalearn=metalearn,
                seed=seed,
                validation_fraction_percent=validation_fraction_percent,
                xu=xu,
                xp=xp,
                treatment=treatment,
                outcome=outcome,
                basename_suffix=basename_suffix,
                save_predictions=save_predictions,
                dataset_predictions_root=dataset_predictions_root,
                outcome_task=outcome_task,
                po_test=po_test,
            )
            results_by_frac[frac].append(result_row)

    if not write_results_csv:
        return {
            frac: pd.DataFrame(results_by_frac[frac]).mean(axis=0, numeric_only=True)
            for frac in train_fracs
        }

    for frac in train_fracs:
        results_ = pd.DataFrame(results_by_frac[frac])
        results_mean_df = pd.DataFrame([results_.mean(axis=0, numeric_only=True)])
        results_out = pd.concat([results_, results_mean_df], axis=0, ignore_index=True)

        out_name = (
            f"{dataset}_{model_name}_k{training_split}_train{train_frac_token[frac]}_{metalearn}_"
            f"seed{seed}"
            f"{basename_suffix}{results_filename_suffix()}"
        )
        results_file_name = path_to_results + "/results_version.csv".replace("version", out_name)
        results_out.to_csv(results_file_name, index=False)

    return None


def run_experiment_simulated(
    *,
    dataset: str,
    X_train: np.ndarray,
    w_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    w_test: np.ndarray,
    y_test: np.ndarray,
    mu0_test: np.ndarray,
    mu1_test: np.ndarray,
    model_name: str,
    training_split: int,
    metalearn: str,
    seed: int,
    validation_fraction_percent: float,
    path_to_predictions: str,
    basename_suffix: str = "",
    save_predictions: bool = True,
) -> Dict[float, pd.Series]:
    """
    Simulated-data runner using a given train/test split (no KFold).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_fracs = [1.0]
    results_by_frac: Dict[float, list] = {frac: [] for frac in train_fracs}

    X_full = np.vstack(
        [np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)]
    )
    w_full = np.concatenate([np.asarray(w_train).reshape(-1), np.asarray(w_test).reshape(-1)])
    y_full = np.concatenate([np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)])

    xu = torch.as_tensor(X_full, dtype=torch.float32, device=device)
    treatment = torch.as_tensor(w_full, dtype=torch.float32, device=device)
    outcome = torch.as_tensor(y_full, dtype=torch.float32, device=device)
    xp = torch.eye(1, device=device)
    train_pool_indices = np.arange(len(X_train), dtype=np.int64)
    test_indices = np.arange(len(X_train), len(X_train) + len(X_test), dtype=np.int64)
    outcome_task = infer_outcome_task(y_full)
    dataset_predictions_root = str(Path(path_to_predictions) / dataset)
    po_test = np.column_stack(
        [np.asarray(mu0_test).reshape(-1), np.asarray(mu1_test).reshape(-1)]
    )

    for frac in train_fracs:
        result_row = frac_experiment(
            frac=frac,
            train_pool_indices=train_pool_indices,
            test_indices=test_indices,
            fold_idx=1,
            dataset=dataset,
            model_name=model_name,
            training_split=training_split,
            metalearn=metalearn,
            seed=seed,
            validation_fraction_percent=validation_fraction_percent,
            xu=xu,
            xp=xp,
            treatment=treatment,
            outcome=outcome,
            basename_suffix=basename_suffix,
            save_predictions=save_predictions,
            dataset_predictions_root=dataset_predictions_root,
            outcome_task=outcome_task,
            po_test=po_test,
        )
        results_by_frac[frac].append(result_row)

    return {
        frac: pd.DataFrame(results_by_frac[frac]).mean(axis=0, numeric_only=True)
        for frac in train_fracs
    }

        


def save_fold_predictions(
    path_to_predictions: str,
    experiment_tag: str,
    fold_idx: int,
    train_frac_token: str,
    test_indices: np.ndarray,
    treatment_test: np.ndarray,
    outcome_test: np.ndarray,
    outcome_task: int,
    pred_c: np.ndarray,
    pred_t: np.ndarray,
    uplift: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write one compressed ``.npz`` per fold under
    ``<path_to_predictions>/<experiment_tag>/``.
    """
    out_dir = Path(path_to_predictions) / experiment_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if metadata is not None:
        meta_path = out_dir / "experiment.json"
        if not meta_path.is_file():
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    payload = {
        "test_indices": np.asarray(test_indices, dtype=np.int64),
        "treatment_test": np.asarray(treatment_test, dtype=np.float64).reshape(-1),
        "outcome_test": np.asarray(outcome_test, dtype=np.float64).reshape(-1),
        "outcome_task": np.int32(outcome_task),
        "pred_c": np.asarray(pred_c, dtype=np.float64).reshape(-1),
        "pred_t": np.asarray(pred_t, dtype=np.float64).reshape(-1),
    }
    if uplift is not None:
        payload["uplift"] = np.asarray(uplift, dtype=np.float64).reshape(-1)

    np.savez_compressed(
        out_dir / f"fold{fold_idx:02d}_train{train_frac_token}.npz",
        **payload,
    )



def infer_outcome_task(y: np.ndarray) -> int:
    """
    Infer meta-learner outcome type from labels.

    Returns 0 if Y is binary in {0, 1} (classification / probability outcomes),
    else 1 (regression).
    """
    flat = np.asarray(y, dtype=np.float64).ravel()
    u = np.unique(flat)
    if u.size == 2 and np.allclose(np.sort(u), [0.0, 1.0], atol=1e-5):
        return 0
    return 1


def subsample_train_pool(pool_indices: np.ndarray, frac: float, fold_idx: int, seed: int) -> np.ndarray:
    if frac == 1.0:
        return pool_indices
    n = max(1, int(len(pool_indices) * frac))
    frac_code = int(frac * 1000)  # 1000, 500, 250
    rng = np.random.default_rng(seed + fold_idx * 1000 + frac_code)
    return rng.choice(pool_indices, size=n, replace=False)


def main(
    dataset="retail",
    model_name="XGB",
    training_split=5,
    metalearn="S",
    seed=0,
    validation_fraction_percent=0.2,
    path_to_results: str = DEFAULT_RESULTS_DIR,
    path_to_predictions: str = DEFAULT_PREDICTIONS_DIR,
    criteo_outcome: str = "conversion",
):
    """
    For each k-fold split, keep the test fold fixed and run on the full training
    pool (train fraction 1.0 only).

    We save a CSV with per-fold results + a final mean row.
    """
    # Make sure top-level output roots exist before any CSV/NPZ writes.
    Path(path_to_results).mkdir(parents=True, exist_ok=True)
    Path(path_to_predictions).mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset in ("ihdp", "acic2016", "twins"):
        train_fracs_sim = [1.0]
        train_frac_token_sim = {1.0: "1"}
        sim_fold_means: Dict[float, list] = {frac: [] for frac in train_fracs_sim}

        if dataset == "ihdp":
            data_path = Path(__file__).resolve().parent.parent.parent / "data"
            n_sims = IHDP_N_SIMULATIONS
            for exp in range(1, n_sims + 1):
                (
                    X_train,
                    w_train,
                    y_train,
                    X_test,
                    w_test,
                    y_test,
                    _mu0_tr,
                    _mu1_tr,
                    mu0_test,
                    mu1_test,
                ) = load_ihdp(data_path, exp)
                fold_means = run_experiment_simulated(
                    dataset=dataset,
                    X_train=X_train,
                    w_train=w_train,
                    y_train=y_train,
                    X_test=X_test,
                    w_test=w_test,
                    y_test=y_test,
                    mu0_test=mu0_test,
                    mu1_test=mu1_test,
                    model_name=model_name,
                    training_split=training_split,
                    metalearn=metalearn,
                    seed=seed,
                    validation_fraction_percent=validation_fraction_percent,
                    path_to_predictions=path_to_predictions,
                    basename_suffix=f"_exp{exp:03d}",
                    save_predictions=True,
                )
                for frac in train_fracs_sim:
                    sim_fold_means[frac].append(fold_means[frac])
        elif dataset == "acic2016":
            # ACIC2016: causallib bundled data; instance 1..10, same seed as rest of main().
            n_sims = ACIC2016_N_SIMULATIONS
            for exp in range(1, n_sims + 1):
                (
                    X_train,
                    w_train,
                    y_train,
                    X_test,
                    w_test,
                    y_test,
                    _mu0_tr,
                    _mu1_tr,
                    mu0_test,
                    mu1_test,
                ) = load_acic2016(exp, seed=seed)
                fold_means = run_experiment_simulated(
                    dataset=dataset,
                    X_train=X_train,
                    w_train=w_train,
                    y_train=y_train,
                    X_test=X_test,
                    w_test=w_test,
                    y_test=y_test,
                    mu0_test=mu0_test,
                    mu1_test=mu1_test,
                    model_name=model_name,
                    training_split=training_split,
                    metalearn=metalearn,
                    seed=seed,
                    validation_fraction_percent=validation_fraction_percent,
                    path_to_predictions=path_to_predictions,
                    basename_suffix=f"_exp{exp:03d}",
                    save_predictions=True,
                )
                for frac in train_fracs_sim:
                    sim_fold_means[frac].append(fold_means[frac])
        else:
            # Twins: catenets-style preprocess; one train/test draw per run (seed drives split + treatment).
            n_sims = 1
            (
                X_train,
                w_train,
                y_train,
                X_test,
                w_test,
                y_test,
                _mu0_tr,
                _mu1_tr,
                mu0_test,
                mu1_test,
            ) = load_twins(seed=seed)
            fold_means = run_experiment_simulated(
                dataset=dataset,
                X_train=X_train,
                w_train=w_train,
                y_train=y_train,
                X_test=X_test,
                w_test=w_test,
                y_test=y_test,
                mu0_test=mu0_test,
                mu1_test=mu1_test,
                model_name=model_name,
                training_split=training_split,
                metalearn=metalearn,
                seed=seed,
                validation_fraction_percent=validation_fraction_percent,
                path_to_predictions=path_to_predictions,
                basename_suffix="",
                save_predictions=True,
            )
            for frac in train_fracs_sim:
                sim_fold_means[frac].append(fold_means[frac])

        print(
            f"{dataset.upper()}: saving cross-simulation average metrics ({n_sims} draws) "
            "to <path_to_results>/results_*.csv"
        )
        for frac in train_fracs_sim:
            stacked = pd.DataFrame(sim_fold_means[frac])
            avg_across_sims = pd.DataFrame([stacked.mean(axis=0, numeric_only=True)])
            out_name = (
                f"{dataset}_{model_name}_k{training_split}_train{train_frac_token_sim[frac]}_{metalearn}_"
                f"seed{seed}"
                f"{results_filename_suffix()}"
            )
            results_file_name = path_to_results + "/results_version.csv".replace("version", out_name)
            avg_across_sims.to_csv(results_file_name, index=False)

        return

    po_full: Optional[np.ndarray] = None
    
    if dataset == "retail":
        retail_csv = ensure_retailhero_preprocessed(
            data_dir=(Path(__file__).resolve().parents[1] / "data")
        )
        retail_df = pd.read_csv(retail_csv)
        x_cols = [
            "age",
            "F",
            "M",
            "U",
            "first_issue_abs_time",
            "first_redeem_abs_time",
            "redeem_delay",
        ]
        xu = torch.as_tensor(retail_df[x_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        treatment = torch.as_tensor(
            retail_df["treatment"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device
        ).view(-1, 1)
        outcome = torch.as_tensor(
            retail_df["outcome"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device
        ).view(-1, 1)
        xp = torch.eye(1, device=device)
    
    elif dataset == "hillstrom_men":
        outcome_col = "visit"
        mens_csv, _ = ensure_hillstrom_preprocessed(
            data_dir=(Path(__file__).resolve().parents[1] / "data")
        )
        csv_path = mens_csv
        X_np, T_np, Y_np = load_hillstrom_csv(csv_path, Y=outcome_col)
        xu = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        treatment = torch.as_tensor(T_np, dtype=torch.float32, device=device).view(-1, 1)
        outcome = torch.as_tensor(Y_np, dtype=torch.float32, device=device).view(-1, 1)
        xp = torch.eye(1, device=device)

    elif dataset == "hillstrom_women":
        outcome_col = "visit"
        _, womens_csv = ensure_hillstrom_preprocessed(
            data_dir=(Path(__file__).resolve().parents[1] / "data")
        )
        csv_path = womens_csv
        X_np, T_np, Y_np = load_hillstrom_csv(csv_path, Y=outcome_col)
        xu = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        treatment = torch.as_tensor(T_np, dtype=torch.float32, device=device).view(-1, 1)
        outcome = torch.as_tensor(Y_np, dtype=torch.float32, device=device).view(-1, 1)
        xp = torch.eye(1, device=device)

    elif dataset == "criteo":
        outcome_col = criteo_outcome
        csv_path = ensure_criteo_preprocessed(
            outcome=outcome_col,
            data_dir=(Path(__file__).resolve().parents[1] / "data"),
        )
        X_np, T_np, Y_np = load_criteo_csv(csv_path, Y=outcome_col)
        xu = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        treatment = torch.as_tensor(T_np, dtype=torch.float32, device=device).view(-1, 1)
        outcome = torch.as_tensor(Y_np, dtype=torch.float32, device=device).view(-1, 1)
        xp = torch.eye(1, device=device)

    elif dataset == "criteo_ite":
        csv_path = ensure_criteo_ite_preprocessed(
            data_dir=(Path(__file__).resolve().parents[1] / "data"),
            seed=seed,
        )
        X_np, T_np, Y_np, po_full = load_criteo_ite_csv(csv_path)
        xu = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        treatment = torch.as_tensor(T_np, dtype=torch.float32, device=device).view(-1, 1)
        outcome = torch.as_tensor(Y_np, dtype=torch.float32, device=device).view(-1, 1)
        xp = torch.eye(1, device=device)

    elif dataset == "nhefs":
        X_np, T_np, Y_np = load_nhefs()
        xu = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        treatment = torch.as_tensor(T_np, dtype=torch.float32, device=device).view(-1, 1)
        outcome = torch.as_tensor(Y_np, dtype=torch.float32, device=device).view(-1, 1)
        xp = torch.eye(1, device=device)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            "Expected retail, hillstrom_men, hillstrom_women, criteo, criteo_ite, nhefs, "
            "or simulated ihdp, acic2016, twins."
        )

    run_experiment(
        dataset=dataset,
        xu=xu,
        treatment=treatment,
        outcome=outcome,
        xp=xp,
        model_name=model_name,
        training_split=training_split,
        metalearn=metalearn,
        seed=seed,
        validation_fraction_percent=validation_fraction_percent,
        path_to_results=path_to_results,
        path_to_predictions=path_to_predictions,
        device=device,
        po_full=po_full,
    )
    

    

def run_from_config(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Support both `dataset` (single) and `datasets` (list) keys.
    datasets = cfg.get("datasets", ["retail"] )
    model_names = cfg.get("model_name", ["XGB"])
    training_splits = cfg.get("training_split", [5])
    metalearns = cfg.get("metalearn", ["S"])

    # Accept either `seeds: [..]` or legacy `seed`.
    if "seeds" in cfg:
        seeds_cfg = cfg.get("seeds", [])
    else:
        seeds_cfg = cfg.get("seed", 0)
    seeds = seeds_cfg if isinstance(seeds_cfg, list) else [seeds_cfg]
    seeds = [int(s) for s in seeds]
    validation_fraction_percent = cfg.get("validation_fraction_percent", 0.2)
    path_to_results = cfg.get("path_to_results", DEFAULT_RESULTS_DIR)
    path_to_predictions = cfg.get("path_to_predictions", DEFAULT_PREDICTIONS_DIR)
    criteo_outcome = cfg.get("criteo_outcome", "conversion")

    # If older configs still contain a legacy semi-supervised flag, we ignore it.
    for dataset, model_name, training_split, metalearn, seed in itertools.product(
        datasets,
        model_names,
        training_splits,
        metalearns,
        seeds,
    ):
        if metalearn == "CEVAE":
            # CEVAE runs are intentionally disabled for now.
            pass
            print("Skipping config combo with disabled metalearn='CEVAE'.")
            continue

        print(
            f"Running dataset={dataset}, model_name={model_name}, training_split={training_split}, "
            f"metalearn={metalearn}, "
            f"seed={seed}"
        )
        main(
            dataset=dataset,
            model_name=model_name,
            training_split=training_split,
            metalearn=metalearn,
            seed=seed,
            validation_fraction_percent=validation_fraction_percent,
            path_to_results=path_to_results,
            path_to_predictions=path_to_predictions,
            criteo_outcome=criteo_outcome,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cate_estimation experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON grid config.")

    # Single-run overrides (used when --config is not provided)
    parser.add_argument("--datasets", type=str, default="retail")
    parser.add_argument("--model_name", type=str, default="XGB")
    parser.add_argument("--training_split", type=int, default=5)
    parser.add_argument("--metalearn", type=str, default="S")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for non-config runs, e.g. 0,1,2. Overrides --seed.",
    )
    parser.add_argument("--validation_fraction_percent", type=float, default=0.2)
    parser.add_argument(
        "--path_to_predictions",
        type=str,
        default=DEFAULT_PREDICTIONS_DIR,
        help="Root directory for per-experiment prediction .npz caches.",
    )
    parser.add_argument(
        "--path_to_results",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for results_*.csv outputs.",
    )
    parser.add_argument(
        "--criteo_outcome",
        type=str,
        default="conversion",
        choices=["conversion", "visit"],
        help="Outcome column for Criteo runs (conversion or visit).",
    )

    args = parser.parse_args()

    if args.config:
        run_from_config(args.config)
    else:
        seeds = (
            [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
            if args.seeds
            else [args.seed]
        )
        for seed in seeds:
            print(
                f"Running dataset={args.datasets}, model_name={args.model_name}, "
                f"training_split={args.training_split}, metalearn={args.metalearn}, "
                f"seed={seed}"
            )
            main(
                dataset=args.datasets,
                model_name=args.model_name,
                training_split=args.training_split,
                metalearn=args.metalearn,
                seed=seed,
                validation_fraction_percent=args.validation_fraction_percent,
                path_to_results=args.path_to_results,
                path_to_predictions=args.path_to_predictions,
                criteo_outcome=args.criteo_outcome,
            )

    
