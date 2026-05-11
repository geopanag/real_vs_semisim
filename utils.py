from pathlib import Path
import os
from urllib.request import urlretrieve

import torch
import numpy as np


import torch.nn.functional as F
import random 

from typing import Callable

from torch.optim import Optimizer
import random 
from torch.optim import Adam
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#from models import BipartiteSAGE_1out, BipartiteSAGE_2out

from catenets.datasets import dataset_ihdp, dataset_twins
from catenets.datasets.network import download_if_needed
from causallib.datasets.data_loader import load_acic16, load_nhefs as _causallib_load_nhefs

import pandas as pd

from typing import Optional, Tuple, Union

from catenets.experiment_utils.torch_metrics import abs_error_ATE, sqrt_PEHE

import numpy as np


CRITEO_FEATURE_COLS = [f"f{i}" for i in range(12)]
CRITEO_RAW_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"

from preprocessing.preprocess_criteo import process_criteo_raw, save_clean
from preprocessing.preprocess_criteo_ite import build_criteo_ite
from preprocessing.preprocess_hillstrom import preprocess as preprocess_hillstrom


import numpy as np

def policy_value_from_tau(tau_hat, y0, y1, threshold=0.0, higher_is_better=True):
    """
    Policy value for the induced treatment rule d(x)=1[tau_hat > threshold].

    Parameters
    ----------
    tau_hat : array-like, shape (n,)
        Predicted treatment effects on the evaluation set.
    y0, y1 : array-like, shape (n,)
        True potential outcomes under control and treatment.
        These must be available, so this is for simulated / semi-simulated data.
    threshold : float, default=0.0
        Treatment threshold. Use 0.0 for the canonical no-cost binary treatment rule.
    higher_is_better : bool, default=True
        If True, larger outcomes are better and we treat when tau_hat > threshold.
        If False, smaller outcomes are better (e.g. loss/risk), so we treat when tau_hat < threshold.

    Returns
    -------
    float
        Mean policy value under the induced treatment rule.
    """
    tau_hat = np.asarray(tau_hat).reshape(-1)
    y0 = np.asarray(y0).reshape(-1)
    y1 = np.asarray(y1).reshape(-1)

    if not (len(tau_hat) == len(y0) == len(y1)):
        raise ValueError("tau_hat, y0, and y1 must have the same length.")

    if higher_is_better:
        d = (tau_hat > threshold).astype(float)
    else:
        d = (tau_hat < threshold).astype(float)

    return np.mean(d * y1 + (1.0 - d) * y0)

def policy_value_from_po(pred_c, pred_t, y0, y1):
    # policy p(x) = 1 if predicted treated outcome exceeds predicted control outcome
    p = (np.asarray(pred_t) - np.asarray(pred_c) > 0).astype(float)
    return np.mean(p * np.asarray(y1) + (1.0 - p) * np.asarray(y0))


def set_seed(seed:int)->None: 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)


def uplift_score(
    prediction: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
    rate: float = 0.2,
) -> float:
    """
    Order samples by predicted uplift and compute uplift-at-rate:
    mean(Y | treated, top-rate) - mean(Y | control, top-rate).
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()

    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    return float(treatment_p - control_p)


def load_hillstrom_csv(path, Y="visit"):
    """
    Load one Hillstrom csv file.
    """
    df = pd.read_csv(path)

    Y = df[Y].values          # or "conversion" or "spend" depending on if the estimate is continuous or discrete
    T = df["treatment"].values

    X = df.drop(columns=["visit", "conversion", "spend", "treatment", "segment"]).values # different versions of treatment and Y

    return X, T, Y



def str2bool(s: str) -> bool:
    """
    Convert common CLI boolean string values to a Python bool.
    """
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y"}



def load_criteo_csv(path, Y: str = "conversion"):
    """
    Load a preprocessed Criteo uplift CSV (e.g. from preprocess_criteo.py).

    Expected columns: f0..f11, treatment, and one of conversion / visit as y_col.
    Default outcome is ``conversion``; pass ``y_col='visit'`` for the visit-labeled file.

    Parameters
    ----------
    path : str or path-like
    y_col : str
        Must match the outcome column in the file ('conversion' or 'visit').

    Returns
    -------
    X : np.ndarray, shape (n, 12)
    T : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
    """
    df = pd.read_csv(path)
    need = CRITEO_FEATURE_COLS + ["treatment", Y]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Available: {list(df.columns)}")

    X = df[CRITEO_FEATURE_COLS].to_numpy(dtype=np.float64)
    T = df["treatment"].to_numpy(dtype=np.int64)
    Y = df[Y].to_numpy(dtype=np.float64)
    return X, T, Y


def ensure_criteo_preprocessed(outcome: str, data_dir: Union[str, Path]) -> Path:
    """
    Ensure preprocessed Criteo file exists for selected outcome.
    Auto-download raw .csv.gz and preprocess if needed.
    """
    if outcome not in {"conversion", "visit"}:
        raise ValueError(f"criteo outcome must be conversion/visit, got {outcome!r}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    clean_csv = data_dir / f"criteo_uplift_{outcome}.csv"
    if clean_csv.is_file():
        return clean_csv

    raw_gz = data_dir / "criteo-research-uplift-v2.1.csv.gz"
    if not raw_gz.is_file():
        print(f"Downloading Criteo raw data to {raw_gz} ...")
        urlretrieve(CRITEO_RAW_URL, raw_gz)

    raw_df = pd.read_csv(raw_gz, compression="gzip")
    work_df = process_criteo_raw(raw_df)
    save_clean(work_df, clean_csv, outcome)
    print(f"Prepared {clean_csv.name}")
    return clean_csv


def ensure_criteo_ite_preprocessed(
    data_dir: Union[str, Path],
    *,
    synthetic_setup: str = "multi_exp",
    ate_real: float = 4.0,
    nb_centers: int = 5,
    std_scaler: float = 1.0,
    seed: int = 0,
) -> Path:
    """
    Ensure semi-synthetic Criteo ITE file exists.
    Auto-download raw .csv.gz and build ``criteo_ite.csv`` if needed.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ite_csv = data_dir / "criteo_ite.csv"
    if ite_csv.is_file():
        return ite_csv

    raw_gz = data_dir / "criteo-research-uplift-v2.1.csv.gz"
    if not raw_gz.is_file():
        print(f"Downloading Criteo raw data to {raw_gz} ...")
        urlretrieve(CRITEO_RAW_URL, raw_gz)

    raw_df = pd.read_csv(raw_gz, compression="gzip")
    out_df = build_criteo_ite(
        raw_df=raw_df,
        synthetic_setup=synthetic_setup,
        ate_real=ate_real,
        nb_centers=nb_centers,
        std_scaler=std_scaler,
        seed=seed,
    )
    out_df.to_csv(ite_csv, index=False)
    print(
        f"Prepared {ite_csv} "
        f"shape={out_df.shape} treated_rate={out_df['treatment'].mean():.4f} tau_mean={out_df['tau'].mean():.4f}"
    )
    return ite_csv


def ensure_hillstrom_preprocessed(data_dir: Union[str, Path]) -> tuple[Path, Path]:
    """
    Ensure Hillstrom men/women processed CSVs exist.

    If missing, fetch raw Hillstrom via sklift, build a raw ``hillstrom.csv``,
    then run ``preprocess_hillstrom.preprocess`` to create:
      - hillstrom_men.csv
      - hillstrom_women.csv
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    men_csv = data_dir / "hillstrom_men.csv"
    women_csv = data_dir / "hillstrom_women.csv"
    if men_csv.is_file() and women_csv.is_file():
        return men_csv, women_csv

    # Backward-compatible rename if older pipelines produced the *_mens/*_womens files.
    old_men_csv = data_dir / "hillstrom_mens.csv"
    old_women_csv = data_dir / "hillstrom_womens.csv"
    if old_men_csv.is_file() and not men_csv.is_file():
        old_men_csv.rename(men_csv)
    if old_women_csv.is_file() and not women_csv.is_file():
        old_women_csv.rename(women_csv)
    if men_csv.is_file() and women_csv.is_file():
        return men_csv, women_csv

    raw_csv = data_dir / "hillstrom.csv"
    if not raw_csv.is_file():
        # Lazy import to avoid hard dependency when Hillstrom is not used.
        from sklift.datasets import fetch_hillstrom  # type: ignore

        ds_visit = fetch_hillstrom(target_col="visit")
        ds_conv = fetch_hillstrom(target_col="conversion")
        ds_spend = fetch_hillstrom(target_col="spend")

        raw_df = pd.DataFrame(ds_visit.data).copy()
        raw_df["segment"] = np.asarray(ds_visit.treatment)
        raw_df["visit"] = np.asarray(ds_visit.target)
        raw_df["conversion"] = np.asarray(ds_conv.target)
        raw_df["spend"] = np.asarray(ds_spend.target)
        raw_df.to_csv(raw_csv, index=False)
        print(f"Prepared {raw_csv.name}")

    preprocess_hillstrom(
        input_csv=raw_csv,
        output_mens=men_csv,
        output_womens=women_csv,
        segment_col="segment",
    )
    print(f"Prepared {men_csv.name}")
    print(f"Prepared {women_csv.name}")
    return men_csv, women_csv


def ensure_retailhero_preprocessed(data_dir: Union[str, Path]) -> Path:
    """
    Ensure a flat RetailHero CSV exists at:
      <data_dir>/retailhero/processed/retail.csv

    If missing, invoke the RetailHero preprocessing pipeline (downloads + processing),
    then build a single table with:
      - features: age, F, M, U, first_issue_abs_time, first_redeem_abs_time, redeem_delay
      - treatment: treatment
      - outcome: outcome
    """
    data_dir = Path(data_dir)
    root = data_dir / "retailhero"
    processed_dir = root / "processed"
    retail_csv = processed_dir / "retail.csv"
    if retail_csv.is_file():
        # Recover from stale/buggy cached files that contain no rows.
        try:
            cached = pd.read_csv(retail_csv)
            if len(cached) > 0:
                return retail_csv
            print(f"Cached RetailHero CSV is empty, rebuilding: {retail_csv.name}")
        except Exception as exc:
            print(f"Could not read cached RetailHero CSV ({retail_csv.name}), rebuilding: {exc}")
        retail_csv.unlink(missing_ok=True)

    root.mkdir(parents=True, exist_ok=True)
    from preprocessing.preprocess_retail import prepare_retail_csv  # type: ignore

    print("Preparing RetailHero dataset...")
    path = prepare_retail_csv(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"RetailHero preparation ran but processed file is missing: {path}"
        )
    print(f"Prepared {path.name}")
    return path


def load_criteo_ite_csv(path: Union[str, Path]):
    """
    Semi-synthetic Criteo ITE file from ``preprocess_criteo_ite.py``.

    Expected columns: f0..f11, treatment, outcome_t0, outcome_t1, outcome_factual,
    outcome_counterfactual, tau.

    Returns
    -------
    X : np.ndarray, shape (n, 12)
    T : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
        Observed outcome (``outcome_factual``).
    po : np.ndarray, shape (n, 2)
        Potential outcomes ``[outcome_t0, outcome_t1]`` for PEHE / policy metrics.
    """
    path = Path(path)
    df = pd.read_csv(path)
    need = CRITEO_FEATURE_COLS + [
        "treatment",
        "outcome_factual",
        "outcome_t0",
        "outcome_t1",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Available: {list(df.columns)}")

    for c in CRITEO_FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["treatment"] = pd.to_numeric(df["treatment"], errors="coerce")
    for c in ("outcome_factual", "outcome_t0", "outcome_t1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=need).copy()
    if not set(df["treatment"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: treatment must be binary 0/1")

    X = df[CRITEO_FEATURE_COLS].to_numpy(dtype=np.float64)
    T = df["treatment"].to_numpy(dtype=np.float64)
    Y = df["outcome_factual"].to_numpy(dtype=np.float64)
    po = np.column_stack(
        [df["outcome_t0"].to_numpy(dtype=np.float64), df["outcome_t1"].to_numpy(dtype=np.float64)]
    )
    return X, T, Y, po


def load_ihdp(data_path: Union[str, Path], exp: int):
    """
    Load one IHDP file.

    Returns
    -------
    X : np.ndarray, shape (n, 25)
    T : np.ndarray, shape (n,)
    Y : np.ndarray, shape (n,)
    """
    data_path = Path(data_path)
    data_train, data_test = dataset_ihdp.load_raw(data_path)

    train_exp = dataset_ihdp.get_one_data_set(data_train, i_exp=exp, get_po=True)
    test_exp  = dataset_ihdp.get_one_data_set(data_test,  i_exp=exp, get_po=True)

    X_train = train_exp["X"]
    w_train = train_exp["w"].reshape(-1)
    y_train = train_exp["y"].reshape(-1)
    mu0_train = train_exp["mu0"].reshape(-1)
    mu1_train = train_exp["mu1"].reshape(-1)

    X_test = test_exp["X"]
    w_test = test_exp["w"].reshape(-1)
    y_test = test_exp["y"].reshape(-1)
    mu0_test = test_exp["mu0"].reshape(-1)
    mu1_test = test_exp["mu1"].reshape(-1)

    return X_train, w_train, y_train, X_test, w_test, y_test, mu0_train, mu1_train, mu0_test, mu1_test




def load_nhefs(
    *,
    restrict: bool = True,
    augment: bool = True,
    onehot: bool = True,
):
    """
    NHEFS smoking-cessation / weight-change cohort (bundled in causallib).

    Same preprocessing defaults as causallib ``load_nhefs`` (Hernán & Robins).
    Returns the full sample for K-fold experiments in ``run_experiment``.

    Returns
    -------
    X : np.ndarray, shape (n, d)
    w : np.ndarray, shape (n,)  treatment qsmk
    y : np.ndarray, shape (n,)  outcome wt82_71
    """
    data = _causallib_load_nhefs(
        raw=False, restrict=restrict, augment=augment, onehot=onehot
    )
    X = np.asarray(data.X, dtype=np.float64)
    w = np.asarray(data.a, dtype=np.float64).reshape(-1)
    y = np.asarray(data.y, dtype=np.float64).reshape(-1)
    return X, w, y


def load_twins(
    *,
    train_ratio: float = 0.8,
    treatment_type: str = "rand",
    seed: int = 42,
    treat_prop: float = 0.5,
    data_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Twins benchmark (catenets ``dataset_twins`` preprocessing).

    Downloads ``Twin_Data.csv.gz`` under catenets' ``datasets/data`` unless
    ``data_path`` points to a directory that already contains it.

    Returns the same layout as ``load_ihdp`` (train/test + potential outcomes).
    """
    base = (
        Path(data_path)
        if data_path is not None
        else Path(os.path.dirname(dataset_twins.__file__)) / "data"
    )
    csv = base / dataset_twins.DATASET
    download_if_needed(csv, http_url=dataset_twins.URL)

    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv(csv, compression="infer")

    cleaned_columns = [col.replace("'", "").replace("’", "") for col in df.columns]
    df.columns = cleaned_columns

    medrisk_list = [
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
    ]
    other_list = ["cigar", "drink", "wtgain", "gestat", "dmeduc", "nprevist"]
    other_list2 = ["pldel", "resstatb"]

    bin_list = ["dmar"] + medrisk_list
    con_list = ["dmage", "mpcb"] + other_list
    cat_list = ["adequacy"] + other_list2

    for feat in medrisk_list:
        df[feat] = df[feat].apply(lambda x, f=feat: df[f].mode()[0] if x in [8, 9] else x)

    for feat in other_list:
        df.loc[df[feat] == 99, feat] = df.loc[df[feat] != 99, feat].mean()

    df_features = df[con_list + bin_list]

    for feat in cat_list:
        df_features = pd.concat(
            [df_features, pd.get_dummies(df[feat], prefix=feat)], axis=1
        )

    feat_list = [
        "dmage",
        "mpcb",
        "cigar",
        "drink",
        "wtgain",
        "gestat",
        "dmeduc",
        "nprevist",
        "dmar",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "adequacy_1",
        "adequacy_2",
        "adequacy_3",
        "pldel_1",
        "pldel_2",
        "pldel_3",
        "pldel_4",
        "pldel_5",
        "resstatb_1",
        "resstatb_2",
        "resstatb_3",
        "resstatb_4",
    ]

    x = np.asarray(df_features[feat_list], dtype=np.float64)
    y0 = np.asarray(df[["outcome(t=0)"]]).reshape((-1,))
    y0 = np.array(y0 < 9999, dtype=np.int64)
    y1 = np.asarray(df[["outcome(t=1)"]]).reshape((-1,))
    y1 = np.array(y1 < 9999, dtype=np.int64)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    n_obs, _ = x.shape

    if treatment_type == "rand":
        prob = np.ones(n_obs) * treat_prop
    elif treatment_type == "logistic":
        coef = np.random.uniform(-0.1, 0.1, size=(x.shape[1], 1))
        prob = 1 / (1 + np.exp(-(x @ coef).ravel()))
    else:
        raise ValueError(f"treatment_type must be 'rand' or 'logistic', got {treatment_type!r}")

    w = np.random.binomial(1, prob).astype(np.int64)
    y = (y1 * w + y0 * (1 - w)).astype(np.float64)

    if not (0 < train_ratio < 1):
        raise ValueError(
            f"train_ratio must be in (0, 1) so train and test are both nonempty, got {train_ratio}"
        )

    idx = np.random.permutation(n_obs)
    n_train = int(train_ratio * n_obs)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train = x[train_idx]
    w_train = w[train_idx].astype(np.float64)
    y_train = y[train_idx]
    mu0_train = y0[train_idx].astype(np.float64)
    mu1_train = y1[train_idx].astype(np.float64)

    X_test = x[test_idx]
    w_test = w[test_idx].astype(np.float64)
    y_test = y[test_idx]
    mu0_test = y0[test_idx].astype(np.float64)
    mu1_test = y1[test_idx].astype(np.float64)

    return (
        X_train,
        w_train,
        y_train,
        X_test,
        w_test,
        y_test,
        mu0_train,
        mu1_train,
        mu0_test,
        mu1_test,
    )


def load_acic2016(
    exp: int,
    *,
    train_size: int = 4000,
    seed: int = 0,
):
    """
    ACIC 2016 from causallib (CSV files ship inside the causallib package). Train/test split.

    Parameters
    ----------
    exp : int
        Passed to ``causallib.datasets.data_loader.load_acic16(instance=exp)``; must be in {1,..,10}.
    train_size : int
        Size of the training split (sklearn ``train_test_split``).
    seed : int
        ``random_state`` for the train/test split.
    """
    if not (1 <= exp <= 10):
        raise ValueError(f"exp must be in 1..10, got {exp}")

    data = load_acic16(instance=exp, raw=False)

    # Causallib returns DataFrame/Series; force numeric float64 so vstack/torch see no object dtype.
    X_df = data.X
    if isinstance(X_df, pd.DataFrame):
        X = np.ascontiguousarray(X_df.to_numpy(dtype=np.float64))
    else:
        X = np.ascontiguousarray(np.asarray(X_df, dtype=np.float64))

    w = np.asarray(data.a, dtype=np.float64).reshape(-1)
    y = np.asarray(data.y, dtype=np.float64).reshape(-1)

    # In causallib, po columns come from mu0/mu1 and are renamed to "0" and "1"
    po = data.po
    mu0 = np.asarray(po["0"], dtype=np.float64).reshape(-1)
    mu1 = np.asarray(po["1"], dtype=np.float64).reshape(-1)

    n = X.shape[0]
    if not (1 <= train_size < n):
        raise ValueError(f"train_size must be in [1, {n-1}], got {train_size}")

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
    )

    X_train = X[train_idx]
    w_train = w[train_idx]
    y_train = y[train_idx]
    mu0_train = mu0[train_idx]
    mu1_train = mu1[train_idx]

    X_test = X[test_idx]
    w_test = w[test_idx]
    y_test = y[test_idx]
    mu0_test = mu0[test_idx]
    mu1_test = mu1[test_idx]

    return (
        X_train,
        w_train,
        y_train,
        X_test,
        w_test,
        y_test,
        mu0_train,
        mu1_train,
        mu0_test,
        mu1_test,
    )



def make_binary_feature(x, 
                    train_indices, 
                    causal_param):
    """
    Define a feature with 2 dims, 0,1 if T=1 and 1,0 if T=0
    """
    t_hat = torch.zeros(x.size(0), 2, dtype=torch.float)
    t_hat[train_indices, causal_param.type(torch.LongTensor)[train_indices]]=1
    return t_hat



def compute_label_prop(xu, 
                        xp, 
                        edge_index_up_current, 
                        train_indices, 
                        causal_param):
    """
    Propagate the causal_param (treatment or outcome) of the training nodes through the bipartite matrix
    """
    sparse_matrix_edge_index = torch.sparse_coo_tensor(
                                edge_index_up_current,
                                torch.ones(edge_index_up_current.shape[1]),
                                (xu.shape[0], xp.shape[0]),
                                dtype=torch.float
                            )

    treatment_n = make_binary_feature(xu, train_indices, causal_param)

    product_treatment_matrix = torch.sparse.mm(sparse_matrix_edge_index.t(), treatment_n.to_sparse())
    lp_neighborhood = torch.sparse.mm(sparse_matrix_edge_index, product_treatment_matrix)
    lp_neighborhood=lp_neighborhood.to_dense().to(xu.device)
    
    min_tn = lp_neighborhood.min(dim=0,keepdim=True).values
    max_tn = lp_neighborhood.max(dim=0,keepdim=True).values
    lp_neighborhood = (lp_neighborhood-min_tn)/(max_tn-min_tn)
    return lp_neighborhood




def rate(
    X,
    y,
    treatment,
    train_indices,
    test_indices,
    pred_c=None,
    pred_t=None,
    tau_pred=None,
    propensity_model=None,
    metric="autoc",
    clip_propensity=1e-3,
    return_curve=False,
):
    """
    Compute RATE on the test set using a propensity model fit on the train fold.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Observed outcomes.
    treatment : array-like
        Binary treatment assignments.
    train_indices, test_indices : array-like
        Train/test split indices.
    pred_c, pred_t : array-like or None
        Predicted mu0 and mu1 on the test set. Optional when ``tau_pred`` is
        provided; if missing/invalid, RATE falls back to a tau-only pseudo-outcome.
    tau_pred : array-like or None
        Predicted CATE/treatment effect on the test set. If provided, this is used
        as the ranking/effect score instead of ``pred_t - pred_c``.
    propensity_model : sklearn-style classifier or None
        Model for P(W=1|X). If None, defaults to LogisticRegression.
    metric : {"qini", "autoc"}
    clip_propensity : float
    return_curve : bool

    Returns
    -------
    rate_value or (rate_value, extras)
    """
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_test = np.asarray(y[test_indices]).reshape(-1)
    t_train = np.asarray(treatment[train_indices]).reshape(-1).astype(int)
    t_test = np.asarray(treatment[test_indices]).reshape(-1).astype(int)

    pred_c_arr = None
    pred_t_arr = None
    has_valid_mu = False
    if pred_c is not None and pred_t is not None:
        pred_c_arr = np.asarray(pred_c).reshape(-1)
        pred_t_arr = np.asarray(pred_t).reshape(-1)
        if len(pred_c_arr) == len(test_indices) and len(pred_t_arr) == len(test_indices):
            has_valid_mu = np.isfinite(pred_c_arr).all() and np.isfinite(pred_t_arr).all()

    if tau_pred is not None:
        tau_pred = np.asarray(tau_pred).reshape(-1)
        if len(tau_pred) != len(test_indices):
            raise ValueError("tau_pred must be predictions on the test set.")
    else:
        if not has_valid_mu:
            raise ValueError("Either tau_pred or finite pred_c/pred_t on test set must be provided.")
        tau_pred = pred_t_arr - pred_c_arr

    if propensity_model is None:
        propensity_model = LogisticRegression(max_iter=1000)

    prop_clf = clone(propensity_model)
    prop_clf.fit(X_train, t_train)

    if hasattr(prop_clf, "predict_proba"):
        e_hat = prop_clf.predict_proba(X_test)[:, 1]
    elif hasattr(prop_clf, "decision_function"):
        logits = prop_clf.decision_function(X_test)
        e_hat = 1.0 / (1.0 + np.exp(-logits))
    else:
        raise ValueError("propensity_model must support predict_proba or decision_function.")

    e_hat = np.clip(e_hat, clip_propensity, 1.0 - clip_propensity)

    if has_valid_mu:
        # DR pseudo-outcome when mu0/mu1 are available.
        gamma = (
            tau_pred
            + t_test * (y_test - pred_t_arr) / e_hat
            - (1 - t_test) * (y_test - pred_c_arr) / (1 - e_hat)
        )
    else:
        # Tau-only fallback pseudo-outcome (AIPW without outcome regressions).
        # This keeps RATE computable for methods that only output tau(x).
        gamma = y_test * (t_test - e_hat) / (e_hat * (1.0 - e_hat))

    # Ranking score
    score = tau_pred
    order = np.argsort(-score)

    gamma_sorted = gamma[order]
    score_sorted = score[order]

    overall_mean = gamma_sorted.mean()
    csum = np.cumsum(gamma_sorted)
    k = np.arange(1, len(gamma_sorted) + 1)
    u = k / len(gamma_sorted)
    toc = csum / k - overall_mean

    if metric == "qini":
        alpha = u
    elif metric == "autoc":
        alpha = np.ones_like(u)
    else:
        raise ValueError("metric must be 'qini' or 'autoc'.")

    rate_value = np.mean(alpha * toc)

    if return_curve:
        return rate_value, {
            "propensity": e_hat,
            "gamma": gamma,
            "score": score,
            "sorted_score": score_sorted,
            "sorted_gamma": gamma_sorted,
            "u": u,
            "toc": toc,
        }

    return rate_value



def uplift_score(prediction: torch.tensor, 
                 treatment: torch.tensor, 
                 target: torch.tensor, 
                 rate=0.2) -> float:
    """
    From https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    Order the samples by the predicted uplift. 
    Calculate the average ground truth outcome of the top rate*100% of the treated and the control samples.
    Subtract the above to get the uplift. 
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()

    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score


def qini_curve(pred_treatment_effects, actual_outcomes, treatment, quantiles=10):
    """
    Computes the QINI curve for non-binary outcomes.
    
    Parameters:
    - pred_treatment_effects: Predicted individual treatment effects (tau_hat).
    - actual_outcomes: Observed actual outcomes (Y).
    - treatment: Binary treatment assignment (W), 1 for treated, 0 for control.
    - quantiles: Number of quantiles (default 10 for deciles).
    
    Returns:
    - qini_values: The cumulative gain values for each quantile.
    - qini_score: Area under the QINI curve.
    """
    # Combine the data into a DataFrame
    data = pd.DataFrame({
        'pred_tau': pred_treatment_effects,
        'Y': actual_outcomes,
        'W': treatment
    })

    # Sort by predicted treatment effect
    data = data.sort_values(by='pred_tau', ascending=False).reset_index(drop=True)
    
    # Compute quantile groups
    data['quantile'] = pd.qcut(data.index, q=quantiles, labels=False)
    
    # Initialize cumulative gain values
    qini_values = []
    cumulative_gain = 0
    
    # Loop over quantiles
    for q in range(quantiles):
        quantile_data = data[data['quantile'] == q]
        
        # Calculate observed ATE for this quantile
        treated = quantile_data[quantile_data['W'] == 1]['Y'].mean()
        control = quantile_data[quantile_data['W'] == 0]['Y'].mean()
        ATE = treated - control
        
        # Cumulative gain is just the sum of ATE across quantiles
        cumulative_gain += ATE
        qini_values.append(cumulative_gain)
    
    # Normalize the QINI values
    qini_values = np.array(qini_values)
    qini_values = qini_values / np.max(np.abs(qini_values))
    
    # QINI score: Area under the QINI curve
    qini_score = np.trapz(qini_values, dx=1/quantiles)
    
    return qini_values, qini_score





def evaluate(
    treatment: np.ndarray,
    outcome: np.ndarray,
    pred_t: np.ndarray,
    pred_c: np.ndarray,
    test_indices: np.ndarray,
    X: Union[np.ndarray, None] = None,
    train_indices: Union[np.ndarray, None] = None,
    po_test: Union[np.ndarray, None] = None,
    uplift: Union[np.ndarray, None] = None,
    has_individual_outcomes: bool = True,
):
    """
    Metrics on the test fold. If ``X`` and ``train_indices`` are given, also computes
    RATE-style summaries (``rate_autoc``, ``rate_qini``) via :func:`rate` (model-based
    propensity on the train fold).
    """

    criterion_eval = outcome_regression_loss_l1

    factual_loss = float("nan")
    if has_individual_outcomes:
        factual_loss = criterion_eval(
            treatment[test_indices],
            torch.tensor(pred_t),
            torch.tensor(pred_c),
            outcome[test_indices],
        ).item()

    treatment_test = treatment[test_indices].numpy()
    outcome_test = outcome[test_indices].numpy()

    if uplift is None:
        uplift_eval = np.asarray(pred_t).reshape(-1) - np.asarray(pred_c).reshape(-1) 
    else:
        uplift_eval = np.asarray(uplift).reshape(-1)
        

    if len(uplift_eval) != len(test_indices):
        raise ValueError("uplift must be predictions on the test set.")

    up40 = uplift_score(uplift_eval, treatment_test, outcome_test, 0.4)
    up20 = uplift_score(uplift_eval, treatment_test, outcome_test, 0.2)
    up10 = uplift_score(uplift_eval, treatment_test, outcome_test, 0.1)

    uplift_diff = abs(
        uplift_eval.mean()
        - (
            outcome_test[treatment_test == 1].mean()
            - outcome_test[treatment_test == 0].mean()
        )
    )

    qini_plot, qini_score = qini_curve(uplift_eval, outcome_test, treatment_test, quantiles=10)

    rate_autoc = float("nan")
    rate_qini = float("nan")
    policy_value_tau = float("nan")
    abs_error_ate = float("nan")
    sqrt_pehe = float("nan")
    if X is not None and train_indices is not None:
        try:
            y_full = outcome.detach().cpu().numpy().reshape(-1)
            t_full = treatment.detach().cpu().numpy().reshape(-1)
            rate_autoc = float(
                rate(
                    X,
                    y_full,
                    t_full,
                    train_indices,
                    test_indices,
                    pred_c,
                    pred_t,
                    tau_pred=uplift_eval,
                    metric="autoc",
                )
            )
            rate_qini = float(
                rate(
                    X,
                    y_full,
                    t_full,
                    train_indices,
                    test_indices,
                    pred_c,
                    pred_t,
                    tau_pred=uplift_eval,
                    metric="qini",
                )
            )
        except Exception as exc:
            print(f"rate() skipped: {exc}")

    if po_test is not None:
        try:
            po_test = np.asarray(po_test, dtype=np.float64)
            if po_test.ndim != 2 or po_test.shape[1] != 2:
                raise ValueError("po_test must have shape (n_test, 2)")
            if len(po_test) != len(test_indices):
                raise ValueError("po_test length must match test set size")

            hat_te = uplift_eval
            policy_value_tau = float(
                policy_value_from_tau(
                    tau_hat=hat_te,
                    y0=po_test[:, 0],
                    y1=po_test[:, 1],
                )
            )
            abs_error_ate = float(abs_error_ATE(po_test, hat_te))
            sqrt_pehe = float(sqrt_PEHE(po_test, hat_te))
        except Exception as exc:
            print(f"simulated PO metrics skipped: {exc}")

    print(f"uplift diff {uplift_diff:.4f}")

    print(
        f"factual mae {factual_loss:.4f} with avg abs value {torch.mean(torch.abs(outcome[test_indices]))}"
    )

    print(
        f"up10 {up10:.4f} up20 {up20:.4f} up40 {up40:.4f} "
        f"rate_autoc {rate_autoc} rate_qini {rate_qini} "
        f"policy_value_from_tau {policy_value_tau} abs_error_ATE {abs_error_ate} sqrt_PEHE {sqrt_pehe}"
    )

    return {
        "up40": up40,
        "up20": up20,
        "up10": up10,
        "factual_loss": factual_loss,
        "uplift_diff": uplift_diff,
        "qini_score": qini_score,
        "rate_autoc": rate_autoc,
        "rate_qini": rate_qini,
        "policy_value_from_tau": policy_value_tau,
        "policy_value_from_po": policy_value_tau,
        "abs_error_ATE": abs_error_ate,
        "sqrt_PEHE": sqrt_pehe,
    }



def learn_node_representations( xu: torch.tensor = None,
                                xp: torch.tensor = None,
                                treatment: torch.tensor = None,
                                outcome: torch.tensor = None,
                                train_indices: np.ndarray = None, 
                                val_indices: np.ndarray = None,
                                test_indices: np.ndarray = None,
                                edge_index_up_current: torch.tensor = None,
                                k: int = 0,
                                conv_layer: str = "sageconv" , 
                                n_hidden: int = 16, 
                                lr:float = 0.01, 
                                l2_reg:float = 5e-4, 
                                dropout:float = 0.2, 
                                no_layers:int = 1, 
                                out_channels:int = 1, 
                                num_epochs:int = 500,
                                patience:int = 100, 
                                print_per_epoch:int = 50):
    pass #TODO: Implement this
    """
    device = xu.device
    criterion_train = outcome_regression_loss_l1 #outcome_regression_loss_l1_one_output
    
    criterion_eval = outcome_regression_loss_l1

    model_file_name = "../../models/"+conv_layer+"_L_"+str(no_layers)+ "_kfold_"+str(k)+ "_model.pt"
    
    edge_index_up_current[1] = edge_index_up_current[1]+ xu.shape[0]

    edge_index_up_current = torch.cat([edge_index_up_current,edge_index_up_current.flip(dims=[0])],dim=1).to(device)
   
    model = BipartiteSAGE_2out(xu.shape[1], xp.shape[1] , n_hidden, out_channels, no_layers, conv_layer, dropout).to(device)
    #model = BipartiteSAGE_1out(xu.shape[1]+1, xp.shape[1] , n_hidden, out_channels, no_layers, conv_layer, dropout).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay = l2_reg)

    # init params
    out1, out0, representations  = model( xu, #torch.cat((xu, treatment.unsqueeze(1)),dim=1), 
                                  xp , edge_index_up_current) 

    train_losses, val_losses = experiment(model, optimizer, num_epochs, train_indices, val_indices, 
                                          edge_index_up_current, treatment, outcome, 
                                           #torch.cat((xu, treatment.unsqueeze(1)),dim=1), 
                                           xu,
                                           xp, model_file_name, 
                                           print_per_epoch, patience, 
                                           criterion_train)

    model = torch.load(model_file_name).to(device)

    model.eval()

    pred_t,pred_c, representations  = model( xu, #torch.cat((xu, treatment.unsqueeze(1)),dim=1), 
                                   xp, edge_index_up_current)

    up40, up20, up10, test_loss = evaluate_gnn(model,  treatment, outcome, test_indices, xu, xp, edge_index_up_current, criterion_eval)
    
    print(f'{up40} {up20} {up10} {test_loss}------------------------------------------')
    #print(treatment[test_indices].min()) 
    #print(treatment[test_indices].max() )

    print(f'gnn eval: {test_loss:.4f} {up10} {up20} {up40} ------------------------------------------')

    return representations
    """


def train(mask: np.ndarray, 
          model:torch.nn.Module, 
          xu: torch.tensor, 
          xp: torch.tensor, 
          edge_index: torch.tensor, 
          treatment: torch.tensor, 
          outcome: torch.tensor,
          optimizer: Optimizer, 
          criterion: Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor] ) -> torch.tensor:
    """
    Trains the model for one epoch.
    """
    model.train()
    optimizer.zero_grad() 

    pred_t, pred_c, representations = model(xu, xp, edge_index)    
    loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])

    #pred, _ = model(xu, xp, edge_index)
    #loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])
    
    loss.backward()  
    optimizer.step() 
    return loss


def test(mask: np.ndarray, 
          model:torch.nn.Module, 
          xu: torch.tensor, 
          xp: torch.tensor, 
          edge_index: torch.tensor, 
          treatment: torch.tensor, 
          outcome: torch.tensor,
          criterion: Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor] ) -> torch.tensor:
    """
    Tests the model. 
    """
    model.eval()
    pred_t, pred_c, representations = model(xu, xp, edge_index)
    loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])

    #pred_t, pred_c =  model(xu, xp, edge_index)
    #loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])
    return loss


def experiment(model:torch.nn.Module, 
               optimizer : Optimizer, 
               num_epochs: int, 
               train_indices: np.ndarray, 
               val_indices: np.ndarray, 
               edge_index : torch.tensor, 
               treatment: torch.tensor, 
               outcome: torch.tensor, 
               xu: torch.tensor , 
               xp: torch.tensor , 
               model_file: str, 
               print_per_epoch: int, 
               patience: int,
               criterion : Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor]) -> (list,list) :
    """
    Trains the model for num_epochs epochs and returns the train and validation losses.
    """
    early_stopping = 0
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    print_per_epoch = 50

    for epoch in range(num_epochs):
        train_loss = train(train_indices, model, xu, xp, edge_index, treatment, outcome, optimizer, criterion)
        val_loss = test(val_indices, model, xu, xp, edge_index, treatment, outcome, criterion)

        train_losses.append(float(train_loss.item())) 
        val_losses.append(float(val_loss.item()))

        if val_loss < best_val_loss:
            early_stopping=0
            best_val_loss = val_loss
            torch.save(model, model_file)
        else:
            early_stopping += 1
            if early_stopping > patience:
                print("early stopping..")
                break
                
        if epoch%print_per_epoch==0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}') 
            
    return train_losses, val_losses



def evaluate_gnn(model:torch.nn.Module,       
             treatment: torch.tensor, 
             outcome: torch.tensor,       
             test_indices: np.ndarray, 
             xu: torch.tensor, 
             xp: torch.tensor, 
             edge_index : torch.tensor,
             criterion) -> (float, float,float):

    """
    Evaluates the model on the test set.
    """

    model.eval()

    mask = test_indices
    pred_t, pred_c, representations = model(xu, xp, edge_index)
    
    #treatment_u = treatment.clone().unsqueeze(1)

    #treatment_u[mask]=0
    #pred_c, _ = model(torch.cat((xu,treatment_u),dim=1), xp, edge_index)#model(torch.cat((xu,treatment_u),dim=1), xp, edge_index)
    #treatment_u[mask]=1
    #pred_t, _ = model(torch.cat((xu,treatment_u),dim=1), xp, edge_index)

    factual_loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask]).item()
    
    #counterfactual_loss =  criterion(1- treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])

    treatment_test = treatment[test_indices].detach().cpu().numpy()
    outcome_test = outcome[test_indices].detach().cpu().numpy()
    pred_t = pred_t.detach().cpu().numpy()
    pred_c = pred_c.detach().cpu().numpy()

    uplift = pred_t[test_indices] - pred_c[test_indices]
    uplift = uplift.squeeze()

    up40 = uplift_score(uplift, treatment_test, outcome_test,0.4)
    up20 = uplift_score(uplift, treatment_test, outcome_test,0.2)
    up10 = uplift_score(uplift, treatment_test, outcome_test,0.1)

    return up10, up20, up40, factual_loss#, counterfactual_loss




def outcome_regression_loss_l1(t_true: torch.tensor,
                               y_treatment_pred: torch.tensor, 
                               y_control_pred: torch.tensor, 
                               y_true: torch.tensor) -> torch.tensor:
    """
    Compute mse for treatment and control output layers using treatment vector for masking out the counterfactual predictions
    """
    loss0 = torch.mean(((1. - t_true) * F.l1_loss(y_control_pred.squeeze(), y_true, reduction='none')) )
    loss1 = torch.mean((t_true *  F.l1_loss(y_treatment_pred.squeeze(), y_true, reduction='none') ))

    return loss0 + loss1


def outcome_regression_loss_l1_one_output(y_pred: torch.tensor, 
                               y_true: torch.tensor) -> torch.tensor:
    """
    Simple MAE loss (used when the treatment is given)
    """
    # loss0 = torch.mean(((1. - t_true) * F.l1_loss(y_control_pred.squeeze(), y_true, reduction='none')) )
    loss = F.l1_loss(y_pred.squeeze(), y_true)

    return loss


def outcome_regression_loss(t_true: torch.tensor,
                            y_treatment_pred: torch.tensor, 
                            y_control_pred: torch.tensor, 
                            y_true: torch.tensor) -> torch.tensor:
    """
    Compute mse for treatment and control output layers using treatment vector for masking out the counterfactual predictions
    Used when the treatment is not given or is not available (run for T=1 and T=0)
    """
    loss0 = torch.mean((1. - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction='none')) 
    loss1 = torch.mean(t_true *  F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction='none') )

    return loss0 + loss1


def binary_treatment_loss(t_true, t_pred):
    """
    Compute cross entropy for propensity score , from Dragonnet
    """
    t_pred = (t_pred + 0.001) / 1.002
    
    return torch.mean(F.binary_cross_entropy(t_pred.squeeze(), t_true))




def outcome_regression_loss_dragnn(t_true: torch.tensor,y_treatment_pred: torch.tensor, y_control_pred: torch.tensor, t_pred: torch.tensor, y_true: torch.tensor):
    """
    Compute mse for treatment and control output layers using treatment vector for masking 
    """
   
    loss0 = torch.mean((1. - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction='none')) 
    loss1 = torch.mean(t_true *  F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction='none') )

    lossT = binary_treatment_loss(t_true.float(), F.sigmoid(t_pred))

    return loss0 + loss1 + lossT




def uplift_score(prediction: torch.tensor, 
                 treatment: torch.tensor, 
                 target: torch.tensor, 
                 rate=0.2) -> float:
    """
    From https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    Order the samples by the predicted uplift. 
    Calculate the average ground truth outcome of the top rate*100% of the treated and the control samples.
    Subtract the above to get the uplift. 
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()

    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score

