from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from causalml.inference.meta import (
    BaseDRRegressor,
    BaseRClassifier,
    BaseRRegressor,
    BaseSClassifier,
    BaseSRegressor,
    BaseTClassifier,
    BaseTRegressor,
    BaseXClassifier,
    BaseXRegressor,
)
#from causalml.inference.nn import CEVAE
from causalml.inference.tf import DragonNet
from causalml.inference.tf.utils import regression_loss
from causalml.inference.tree import CausalRandomForestRegressor, UpliftTreeClassifier
from causalml.inference.tree.causal.causaltree import CausalTreeRegressor
from causalml.propensity import ElasticNetPropensityModel

from utils import uplift_score


from cfr import CFR

# Lightweight inner-optimization for base learners.
TUNE_BASE_MODELS = True
INNER_CV = 3

DEFAULT_CFR_CFG = {
    "alpha": 10**6,
    "lr": 1e-3,
    "wd": 0.5,
    "sig": 0.1,
    "epochs": 1000,
    "ipm_type": "mmd_lin",
    "repnet_num_layers": 3,
    "repnet_hidden_dim": 48,
    "repnet_out_dim": 48,
    "repnet_dropout": 0.145,
    "outnet_num_layers": 3,
    "outnet_hidden_dim": 32,
    "outnet_dropout": 0.145,
    "gamma": 0.97,
    "split_outnet": True,
}


def results_filename_suffix() -> str:
    """Token appended to ``results_*.csv`` basenames when base-learner HPO is on."""
    return "_optimized" if TUNE_BASE_MODELS else ""


def make_regressor(model_name: str, tune: bool = TUNE_BASE_MODELS, random_state: int = 42):
    if model_name == "LR":
        return RidgeCV(alphas=np.logspace(-4, 4, 9))
    if model_name == "RF":
        base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        if not tune:
            return base
        return GridSearchCV(
            estimator=base,
            param_grid={"n_estimators": [100, 300], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 5, 10]},
            cv=INNER_CV,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    if model_name == "XGB":
        base = XGBRegressor(
            random_state=random_state, n_jobs=-1, tree_method="hist", eval_metric="rmse"
        )
        if not tune:
            return base
        return GridSearchCV(
            estimator=base,
            param_grid={"n_estimators": [100, 300], "max_depth": [3, 6], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
            cv=INNER_CV,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    if model_name == "NN":
        base = MLPRegressor(
            max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=random_state
        )
        if not tune:
            return MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=random_state
            )
        return GridSearchCV(
            estimator=base,
            param_grid={"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 5e-4]},
            cv=INNER_CV,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    raise ValueError(f"Unknown regressor model_name={model_name!r}")


def make_classifier(model_name: str, tune: bool = TUNE_BASE_MODELS, random_state: int = 42):
    if model_name == "LR":
        if tune:
            return LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10], cv=INNER_CV, max_iter=2000, n_jobs=-1)
        return LogisticRegression(max_iter=2000)

    if model_name == "XGB":
        base = XGBClassifier(
            random_state=random_state, n_jobs=-1, tree_method="hist", eval_metric="logloss"
        )
        if not tune:
            return base
        return GridSearchCV(
            estimator=base,
            param_grid={"n_estimators": [100, 300], "max_depth": [3, 6], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
            cv=INNER_CV,
            scoring="neg_log_loss",
            n_jobs=-1,
        )
    if model_name == "NN":
        base = MLPClassifier(
            max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=random_state
        )
        if not tune:
            return MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=random_state
            )
        return GridSearchCV(
            estimator=base,
            param_grid={"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 5e-3]},
            cv=INNER_CV,
            scoring="neg_log_loss",
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier model_name={model_name!r}")


# ---------------------------------------------------------------------
# Simple meta-learners returning potential outcomes (y0_hat, y1_hat)
# ---------------------------------------------------------------------


def run_s_learner(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    task: int = 1,
):
    """
    S-learner: single model with treatment as an extra feature.
    Returns (pred_c, pred_t) for test_indices.
    """
    Xt = np.hstack((X, treatment_np))
    if task == 0:
        model = make_classifier(model_name)
        model.fit(Xt[train_indices], y[train_indices].astype(int).ravel())
        zcol = np.zeros((len(test_indices), 1))
        ocol = np.ones((len(test_indices), 1))
        pred_c = model.predict_proba(np.hstack((X[test_indices], zcol)))[:, 1]
        pred_t = model.predict_proba(np.hstack((X[test_indices], ocol)))[:, 1]
        return pred_c, pred_t

    model = make_regressor(model_name)
    model.fit(Xt[train_indices], y[train_indices])

    pred_c = model.predict(np.hstack((X[test_indices], np.zeros((len(test_indices), 1)))))
    pred_t = model.predict(np.hstack((X[test_indices], np.ones((len(test_indices), 1)))))
    return pred_c, pred_t

    

def run_t_learner(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    task: int = 1,
):
    """
    T-learner: separate models for treated and control.
    Returns (pred_c, pred_t) for test_indices.
    """
    x_train = X[train_indices]
    y_train = y[train_indices]
    t_train = treatment_np[train_indices].squeeze()

    if task == 0:
        model_treatment = make_classifier(model_name)
        model_treatment.fit(x_train[t_train == 1, :], y_train[t_train == 1].astype(int).ravel())
        pred_t = model_treatment.predict_proba(X[test_indices])[:, 1]

        model_control = make_classifier(model_name)
        model_control.fit(x_train[t_train == 0, :], y_train[t_train == 0].astype(int).ravel())
        pred_c = model_control.predict_proba(X[test_indices])[:, 1]
        return pred_c, pred_t

    model_treatment = make_regressor(model_name)
    model_treatment.fit(x_train[t_train == 1, :], y_train[t_train == 1])
    pred_t = model_treatment.predict(X[test_indices])

    model_control = make_regressor(model_name)
    model_control.fit(x_train[t_train == 0, :], y_train[t_train == 0])
    pred_c = model_control.predict(X[test_indices])

    return pred_c, pred_t

   


def run_x_learner(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    task: int = 1,
):
    """
    X-learner.

    Returns:
        pred_c, pred_t, te_hat
    where ``te_hat`` is the model's direct treatment-effect estimate on test_indices.
    ``pred_c``/``pred_t`` are still returned for metrics that explicitly require
    potential outcomes.
    """
    x_train = X[train_indices]
    y_train = y[train_indices]
    t_train = treatment_np[train_indices].squeeze()

    # Stage 1: outcome models
    model_mu1 = make_regressor(model_name)
    model_mu0 = make_regressor(model_name)
    model_mu1.fit(x_train[t_train == 1, :], y_train[t_train == 1])
    model_mu0.fit(x_train[t_train == 0, :], y_train[t_train == 0])

    mu1_train = model_mu1.predict(x_train)
    mu0_train = model_mu0.predict(x_train)

    # Stage 2: imputed effects
    d1 = y_train[t_train == 1] - mu0_train[t_train == 1]
    d0 = mu1_train[t_train == 0] - y_train[t_train == 0]

    # Stage 3: effect models
    model_tau1 = make_regressor(model_name)
    model_tau0 = make_regressor(model_name)
    model_tau1.fit(x_train[t_train == 1, :], d1)
    model_tau0.fit(x_train[t_train == 0, :], d0)

    tau1_hat_test = model_tau1.predict(X[test_indices])
    tau0_hat_test = model_tau0.predict(X[test_indices])

    # Stage 4: propensity
    prop_model = ElasticNetPropensityModel()
    prop_model.fit(x_train, t_train)
    e_hat_test = _predict_propensity(prop_model, X[test_indices])

    tau_hat_test = (1.0 - e_hat_test) * tau1_hat_test + e_hat_test * tau0_hat_test

    # Convert to potential outcomes for compatibility with evaluate(...)
    mu0_test = model_mu0.predict(X[test_indices])
    pred_c = mu0_test
    pred_t = mu0_test + tau_hat_test

    return pred_c, pred_t, tau_hat_test


def run_dr_learner(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    eps: float = 1e-6,
    task: int = 1,
):
    """
    Doubly-robust learner returning (pred_c, pred_t, te_hat) on test_indices.
    """
    x_train = X[train_indices]
    y_train = y[train_indices]
    t_train = treatment_np[train_indices].squeeze().astype(float)

    model_mu1 = make_regressor(model_name)
    model_mu0 = make_regressor(model_name)
    model_mu1.fit(x_train[t_train == 1], y_train[t_train == 1])
    model_mu0.fit(x_train[t_train == 0], y_train[t_train == 0])

    mu1_hat_train = model_mu1.predict(x_train)
    mu0_hat_train = model_mu0.predict(x_train)
    mu1_hat_test = model_mu1.predict(X[test_indices])
    mu0_hat_test = model_mu0.predict(X[test_indices])

    prop_model = ElasticNetPropensityModel()
    prop_model.fit(x_train, t_train)
    e_hat_train = np.clip(_predict_propensity(prop_model, x_train), eps, 1 - eps)
    e_hat_test = np.clip(_predict_propensity(prop_model, X[test_indices]), eps, 1 - eps)

    dr_pseudo = (
        mu1_hat_train
        - mu0_hat_train
        + t_train * (y_train - mu1_hat_train) / e_hat_train
        - (1.0 - t_train) * (y_train - mu0_hat_train) / (1.0 - e_hat_train)
    )

    model_tau = make_regressor(model_name)
    model_tau.fit(x_train, dr_pseudo)
    tau_hat_test = model_tau.predict(X[test_indices])

    pred_c = mu0_hat_test
    pred_t = mu0_hat_test + tau_hat_test
    return pred_c, pred_t, tau_hat_test


def _predict_propensity(prop_model, x):
    """
    Return P(T=1|X) across causalml propensity API variants.
    """
    if hasattr(prop_model, "predict_proba"):
        proba = np.asarray(prop_model.predict_proba(x))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(prop_model, "predict"):
        return np.asarray(prop_model.predict(x)).reshape(-1)
    raise AttributeError("Propensity model exposes neither predict_proba nor predict")


def run_cfr(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    task: int = 1,
    alpha: Optional[float] = None,
):
    """
    Counterfactual regression network (CFRNet).

    Imports ``CFR`` from the local package ``cfr`` (``cate_estimation/cfr/``), which uses relative imports so it does not clash with top-level ``utils``.
    Returns ``(pred_c, pred_t, tau_hat)`` on ``test_indices`` with
    ``tau_hat = pred_t - pred_c`` from separate potential-outcome forward passes.

    ``model_name`` is ignored (fixed MLP config). Binary outcomes (``task==0``) use float 0/1.
    ``alpha`` optionally overrides ``DEFAULT_CFR_CFG["alpha"]``. Setting ``alpha=0``
    yields TARNet behavior (no IPM penalty).
    """
    del model_name
    x_train = np.asarray(X[train_indices], dtype=np.float64)
    y_train = np.asarray(y[train_indices], dtype=np.float64).reshape(-1)
    t_train = np.asarray(treatment_np[train_indices], dtype=np.float64).reshape(-1, 1)
    x_test = np.asarray(X[test_indices], dtype=np.float64)
    y_test = np.asarray(y[test_indices], dtype=np.float64).reshape(-1)
    t_test = np.asarray(treatment_np[test_indices], dtype=np.float64).reshape(-1, 1)

    class _DataSet(torch.utils.data.Dataset):
        def __init__(self, x, yy, zz):
            self.x = x
            self.y = yy
            self.z = zz

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            return self.x[index, :], self.y[index], self.z[index, :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train = len(x_train)
    batch_size = max(1, min(50, n_train))
    drop_last = n_train > batch_size
    dataset = _DataSet(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(t_train, dtype=torch.float32),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    cfg = dict(DEFAULT_CFR_CFG)
    if alpha is not None:
        cfg["alpha"] = float(alpha)

    model = CFR(in_dim=x_train.shape[1], out_dim=1, cfg=cfg).to(device)
    model.fit(
        dataloader,
        x_train,
        y_train,
        t_train,
        x_test,
        y_test,
        t_test,
        device,
    )

    x_te = torch.tensor(x_test, dtype=torch.float32, device=device)
    n_te = x_te.shape[0]
    t0 = torch.zeros(n_te, 1, device=device)
    t1 = torch.ones(n_te, 1, device=device)
    mu0 = model.forward(x_te, t0).detach().cpu().numpy().reshape(-1)
    mu1 = model.forward(x_te, t1).detach().cpu().numpy().reshape(-1)
    tau_hat = mu1 - mu0
    return mu0, mu1, tau_hat


def run_causal_forest(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    task: int = 1,
):
    """
    Causal random forest returning test-set CATE only.

    Returns ``(pred_c, pred_t, tau_hat)`` where ``pred_c`` and ``pred_t`` are NaN
    placeholders because this learner predicts treatment effects directly.
    """
    del model_name, task
    x_train = np.asarray(X)[train_indices]
    y_train = np.asarray(y)[train_indices].reshape(-1)
    t_train = np.asarray(treatment_np)[train_indices].reshape(-1).astype(int)
    x_test = np.asarray(X)[test_indices]

    model = CausalRandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X=x_train, treatment=t_train, y=y_train)

    tau_hat = np.asarray(model.predict(x_test)).reshape(-1)
    n_test = len(test_indices)
    pred_c = np.full(n_test, np.nan, dtype=np.float64)
    pred_t = np.full(n_test, np.nan, dtype=np.float64)
    return pred_c, pred_t, tau_hat


def run_r_learner(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    eps: float = 1e-6,
    task: int = 1,
):
    """
    R-learner returning (pred_c, pred_t, te_hat) on test_indices.

    task:
        0 = binary outcome
        1 = continuous outcome
    """
    x_train = X[train_indices]
    x_test = X[test_indices]
    y_train = np.asarray(y[train_indices]).reshape(-1)
    t_train = np.asarray(treatment_np[train_indices]).reshape(-1).astype(float)

    # m(x) = E[Y|X]
    if task == 0:
        model_m = make_classifier(model_name)
        model_m.fit(x_train, y_train.astype(int))
        m_hat_train = np.asarray(model_m.predict_proba(x_train)[:, 1]).reshape(-1)
        m_hat_test = np.asarray(model_m.predict_proba(x_test)[:, 1]).reshape(-1)
    else:
        model_m = make_regressor(model_name)
        model_m.fit(x_train, y_train)
        m_hat_train = np.asarray(model_m.predict(x_train)).reshape(-1)
        m_hat_test = np.asarray(model_m.predict(x_test)).reshape(-1)

    # e(x) = P(T=1|X)
    prop_model = ElasticNetPropensityModel()
    prop_model.fit(x_train, t_train.astype(int))
    e_hat_train = np.clip(
        np.asarray(_predict_propensity(prop_model, x_train)).reshape(-1),
        eps,
        1.0 - eps,
    )
    e_hat_test = np.clip(
        np.asarray(_predict_propensity(prop_model, x_test)).reshape(-1),
        eps,
        1.0 - eps,
    )

    # Residualization
    y_res = y_train - m_hat_train
    t_res = t_train - e_hat_train

    keep = np.abs(t_res) > eps
    x_r_train = x_train[keep]
    pseudo_outcome = y_res[keep] / t_res[keep]
    sample_weight = t_res[keep] ** 2

    # tau-model should be a regressor
    model_tau = make_regressor(model_name)
    try:
        model_tau.fit(x_r_train, pseudo_outcome, sample_weight=sample_weight)
    except TypeError:
        model_tau.fit(x_r_train, pseudo_outcome)

    tau_hat_test = np.asarray(model_tau.predict(x_test)).reshape(-1)

    # Recover mu0, mu1 from m, e, tau
    pred_c = m_hat_test - e_hat_test * tau_hat_test
    pred_t = m_hat_test + (1.0 - e_hat_test) * tau_hat_test

    if task == 0:
        pred_c = np.clip(pred_c, 0.0, 1.0)
        pred_t = np.clip(pred_t, 0.0, 1.0)

    return pred_c, pred_t, tau_hat_test



def baseline_mu0(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    treatment_np: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    task: int = 1,
) -> np.ndarray:
    """
    Fit a baseline control outcome model on training controls and predict mu0(x) on test.
    Used to convert causalml's tau/uplift predictions into (pred_c, pred_t).

    task=1 (regression): predict E[Y|X,T=0].
    task=0 (binary): predict P(Y=1|X,T=0) via predict_proba.
    """
    x_train = X[train_indices]
    y_train = y[train_indices]
    t_train = treatment_np[train_indices].squeeze()
    x_ctrl = x_train[t_train == 0, :]
    y_ctrl = y_train[t_train == 0].ravel()

    if task == 0:
        model_mu0 = make_classifier(model_name)
        model_mu0.fit(x_ctrl, y_ctrl.astype(int))
        proba = model_mu0.predict_proba(X[test_indices])
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]

    model_mu0 = make_regressor(model_name)
    model_mu0.fit(x_ctrl, y_ctrl)
    return model_mu0.predict(X[test_indices])

def uplift_to_potential_outcomes(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    treatment_np: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    uplift: np.ndarray,
    task: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an uplift/tau prediction into (pred_c, pred_t) for evaluate(...).

    Uses a baseline control outcome model mu0(x) fitted on training controls:
      pred_c = mu0(x)
      pred_t = mu0(x) + uplift(x)

    NOTE:
    This is a temporary adapter to allow methods that only output uplift/tau
    to be evaluated by code paths that expect potential outcome predictions.
    For causal models like uplift trees and CEVAE, mu0 is not produced by the
    model itself, so any choice of mu0 is an additional modeling assumption.
    We plan to refactor evaluation so uplift-only methods are scored without
    manufacturing pred_c/pred_t.
    """
    mu0_test = baseline_mu0(
        model_name, X, y, treatment_np, train_indices, test_indices, task=task
    )
    return mu0_test, mu0_test + uplift


# ---------------------------------------------------------------------
# CausalML benchmark wrappers (NOT USED NOW)
# ---------------------------------------------------------------------


def test_causalml(
    confounders: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    k: int,
    task: int = 0,
    causal_model_type: str = "X",
    model_out: str = "XGB",
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Test a causalml model in K-fold CV.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    results = []
    for train_indices, test_indices in kf.split(confounders):
        # keep the original convention used in this repo
        test_indices, train_indices = train_indices, test_indices

        up40, up20 = causalml_run(
            confounders[train_indices],
            outcome[train_indices],
            treatment[train_indices],
            confounders[test_indices],
            outcome[test_indices],
            treatment[test_indices],
            task,
            causal_model_type,
            model_out,
        )
        results.append((up40, up20))
    return pd.DataFrame(results)


def causalml_run(
    confounders_train: np.ndarray,
    outcome_train: np.ndarray,
    treatment_train: np.ndarray,
    confounders_test: np.ndarray,
    outcome_test: np.ndarray,
    treatment_test: np.ndarray,
    task: int = 0,
    causal_model_type: str = "X",
    model_class: str = "XGB",
    model_regr: str = "XGBR",
    total_uplift: bool = False,
) -> tuple[float, float]:
    """
    Run a causalml model and return uplift@40% and uplift@20%.
    """
    dic_mod = {
        "XGB": XGBClassifier,
        "LR": LogisticRegression,
        "XGBR": XGBRegressor,
        "NN": MLPClassifier,
        "NNR": MLPRegressor,
    }

    if causal_model_type == "S":
        learner = (
            BaseSClassifier(learner=dic_mod[model_class]())
            if task == 0
            else BaseSRegressor(learner=dic_mod[model_regr]())
        )
        learner.fit(X=confounders_train, y=outcome_train, treatment=treatment_train)
        uplift = learner.predict(
            X=np.vstack([confounders_train, confounders_test])
            if total_uplift
            else confounders_test,
            treatment=np.hstack([treatment_train, treatment_test])
            if total_uplift
            else treatment_test,
        ).squeeze()

    elif causal_model_type == "T":
        learner = (
            BaseTClassifier(learner=dic_mod[model_class]())
            if task == 0
            else BaseTRegressor(learner=dic_mod[model_regr]())
        )
        learner.fit(X=confounders_train, y=outcome_train, treatment=treatment_train)
        uplift = learner.predict(
            X=np.vstack([confounders_train, confounders_test])
            if total_uplift
            else confounders_test,
            treatment=np.hstack([treatment_train, treatment_test])
            if total_uplift
            else treatment_test,
        ).squeeze()

    elif causal_model_type == "X":
        propensity_model = ElasticNetPropensityModel()
        propensity_model.fit(X=confounders_train, y=treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        learner = (
            BaseXClassifier(
                outcome_learner=dic_mod[model_class](),
                effect_learner=dic_mod[model_regr](),
            )
            if task == 0
            else BaseXRegressor(learner=dic_mod[model_regr]())
        )
        learner.fit(X=confounders_train, y=outcome_train, treatment=treatment_train, p=p_train)
        uplift = learner.predict(
            X=np.vstack([confounders_train, confounders_test]) if total_uplift else confounders_test,
            treatment=np.hstack([treatment_train, treatment_test]) if total_uplift else treatment_test,
            p=np.hstack([p_train, p_test]) if total_uplift else p_test,
        ).squeeze()

    elif causal_model_type == "R":
        propensity_model = ElasticNetPropensityModel()
        propensity_model.fit(X=confounders_train, y=treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        learner = (
            BaseRClassifier(
                outcome_learner=dic_mod[model_class](),
                effect_learner=dic_mod[model_regr](),
            )
            if task == 0
            else BaseRRegressor(learner=dic_mod[model_regr]())
        )
        learner.fit(X=confounders_train, y=outcome_train, treatment=treatment_train, p=p_train)
        uplift = learner.predict(
            X=np.vstack([confounders_train, confounders_test]) if total_uplift else confounders_test,
            treatment=np.hstack([treatment_train, treatment_test]) if total_uplift else treatment_test,
            p=np.hstack([p_train, p_test]) if total_uplift else p_test,
        ).squeeze()

    elif causal_model_type == "D":
        propensity_model = ElasticNetPropensityModel()
        propensity_model.fit(X=confounders_train, y=treatment_train)
        p_train = propensity_model.predict(X=confounders_train)
        p_test = propensity_model.predict(X=confounders_test)

        learner = BaseDRRegressor(
            learner=dic_mod[model_class]() if task == 0 else dic_mod[model_regr](),
            treatment_effect_learner=dic_mod[model_regr](),
        )
        learner.fit(X=confounders_train, y=outcome_train, treatment=treatment_train, p=p_train)
        uplift = learner.predict(
            X=np.vstack([confounders_train, confounders_test]) if total_uplift else confounders_test,
            treatment=np.hstack([treatment_train, treatment_test]) if total_uplift else treatment_test,
            p=np.hstack([p_train, p_test]) if total_uplift else p_test,
        ).squeeze()

    elif causal_model_type == "Tree":
        learner = UpliftTreeClassifier(control_name="0") if task == 0 else CausalTreeRegressor(control_name="0")
        X_train = np.hstack((treatment_train.reshape(-1, 1), confounders_train))
        X_test = np.hstack((treatment_test.reshape(-1, 1), confounders_test))
        learner.fit(X=X_train, treatment=treatment_train.astype(str), y=outcome_train)
        uplift = learner.predict(X=np.vstack([X_train, X_test]).squeeze() if total_uplift else X_test).squeeze()

    elif causal_model_type == "Dragon":
        learner = DragonNet() if task == 0 else DragonNet(loss_func=regression_loss)
        learner.fit(X=confounders_train, treatment=treatment_train, y=outcome_train.astype(np.float32))
        uplift = learner.predict(
            X=np.vstack([confounders_train, confounders_test]) if total_uplift else confounders_test,
            treatment=np.hstack([treatment_train, treatment_test]) if total_uplift else treatment_test,
        )
        uplift = (uplift[:, 1] - uplift[:, 0]).squeeze()

    elif causal_model_type == "CEVAE":
        pass
        #learner = CEVAE()
        #learner.fit(
        #    X=np.asarray(confounders_train, dtype=np.float32),
        #    treatment=np.asarray(treatment_train, dtype=np.float32),
        #    y=np.asarray(outcome_train, dtype=np.float32),
        #)
        #uplift = learner.predict(
        #    X=np.vstack([confounders_train, confounders_test]) if total_uplift else confounders_test,
        #    treatment=np.hstack([treatment_train, treatment_test]) if total_uplift else treatment_test,
        #).squeeze()

    else:
        raise ValueError(f"Unknown causal_model_type={causal_model_type!r}")

    if total_uplift:
        treatment_all = np.hstack([treatment_train, treatment_test])
        outcome_all = np.hstack([outcome_train, outcome_test])
        score40 = uplift_score(uplift, treatment_all, outcome_all, rate=0.4)
        score20 = uplift_score(uplift, treatment_all, outcome_all, rate=0.2)
    else:
        score40 = uplift_score(uplift, treatment_test, outcome_test, rate=0.4)
        score20 = uplift_score(uplift, treatment_test, outcome_test, rate=0.2)

    return score40, score20


def run_causal_tree(
    confounders_train: np.ndarray,
    outcome_train: np.ndarray,
    treatment_train: np.ndarray,
    confounders_test: np.ndarray,
    outcome_test: np.ndarray,
    treatment_test: np.ndarray,
    task: int = 0,
    total_uplift: bool = False,
) -> tuple[float, float, np.ndarray]:
    """
    Convenience wrapper for the causal tree block used in causalml_run.

    Returns: (uplift@40, uplift@20, uplift_vector)
    """
    learner = (
        UpliftTreeClassifier(control_name="0")
        if task == 0
        else CausalTreeRegressor(control_name="0")
    )
    X_train = np.hstack((treatment_train.reshape(-1, 1), confounders_train))
    X_test = np.hstack((treatment_test.reshape(-1, 1), confounders_test))
    learner.fit(X=X_train, treatment=treatment_train.astype(str), y=outcome_train)

    uplift = learner.predict(
        X=np.vstack([X_train, X_test]).squeeze() if total_uplift else X_test
    ).squeeze()

    if total_uplift:
        treatment_all = np.hstack([treatment_train, treatment_test])
        outcome_all = np.hstack([outcome_train, outcome_test])
        score40 = uplift_score(uplift, treatment_all, outcome_all, rate=0.4)
        score20 = uplift_score(uplift, treatment_all, outcome_all, rate=0.2)
    else:
        score40 = uplift_score(uplift, treatment_test, outcome_test, rate=0.4)
        score20 = uplift_score(uplift, treatment_test, outcome_test, rate=0.2)

    return score40, score20, uplift


def run_dragonnet(
    model_name,
    X,
    y,
    treatment_np,
    train_indices,
    test_indices,
    task: int = 1,
):
    """
    TensorFlow DragonNet (causalml): potential outcomes on the test fold.

    Same calling convention as ``run_s_learner`` / ``run_dr_learner``. Ignores
    ``model_name`` (fixed TF architecture).

    Returns ``(pred_c, pred_t, tau_hat)`` on ``test_indices`` with
    ``tau_hat = pred_t - pred_c`` (columns of ``predict``: control / treated heads).

    See also ``causalml_run`` Dragon branch (lines ~887–894).
    """
    del model_name

    confounders_train = np.asarray(X[train_indices], dtype=np.float64)
    outcome_train = np.asarray(y[train_indices], dtype=np.float64).reshape(-1)
    treatment_train = treatment_np[train_indices].squeeze()
    confounders_test = np.asarray(X[test_indices], dtype=np.float64)
    outcome_test = y[test_indices]
    treatment_test = treatment_np[test_indices].squeeze()

    # DragonNet can become numerically unstable on some ACIC regression runs.
    # Stabilize training by standardizing features (all tasks) and outcomes (task=1).
    x_mean = confounders_train.mean(axis=0, keepdims=True)
    x_std = confounders_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    confounders_train_n = (confounders_train - x_mean) / x_std
    confounders_test_n = (confounders_test - x_mean) / x_std

    y_train_for_fit = outcome_train.astype(np.float32)
    y_mean = 0.0
    y_std = 1.0
    if task == 1:
        y_mean = float(np.mean(outcome_train))
        y_std_raw = float(np.std(outcome_train))
        y_std = y_std_raw if y_std_raw > 1e-8 else 1.0
        y_train_for_fit = ((outcome_train - y_mean) / y_std).astype(np.float32)

    learner = DragonNet() if task == 0 else DragonNet(loss_func=regression_loss)
    learner.fit(
        X=confounders_train_n,
        treatment=treatment_train,
        y=y_train_for_fit,
    )
    y_hat = learner.predict(X=confounders_test_n, treatment=treatment_test)
    pred_c = np.asarray(y_hat[:, 0], dtype=np.float64).reshape(-1)
    pred_t = np.asarray(y_hat[:, 1], dtype=np.float64).reshape(-1)
    if task == 1:
        pred_c = pred_c * y_std + y_mean
        pred_t = pred_t * y_std + y_mean
    tau_hat = pred_t - pred_c
    if not (np.isfinite(pred_c).all() and np.isfinite(pred_t).all() and np.isfinite(tau_hat).all()):
        raise ValueError("DragonNet produced non-finite predictions.")
    return pred_c, pred_t, tau_hat

"""
def run_cevae(
    confounders_train: np.ndarray,
    outcome_train: np.ndarray,
    treatment_train: np.ndarray,
    confounders_test: np.ndarray,
    outcome_test: np.ndarray,
    treatment_test: np.ndarray,
    task: int = 0,
    total_uplift: bool = False,
) -> tuple[float, float, np.ndarray]:
   
    _ = task
    learner = CEVAE()
    learner.fit(
        X=np.asarray(confounders_train, dtype=np.float32),
        treatment=np.asarray(treatment_train, dtype=np.float32),
        y=np.asarray(outcome_train, dtype=np.float32),
    )
    uplift = learner.predict(
        X=np.vstack([confounders_train, confounders_test])
        if total_uplift
        else confounders_test,
        treatment=np.hstack([treatment_train, treatment_test])
        if total_uplift
        else treatment_test,
    ).squeeze()

    if total_uplift:
        treatment_all = np.hstack([treatment_train, treatment_test])
        outcome_all = np.hstack([outcome_train, outcome_test])
        score40 = uplift_score(uplift, treatment_all, outcome_all, rate=0.4)
        score20 = uplift_score(uplift, treatment_all, outcome_all, rate=0.2)
    else:
        score40 = uplift_score(uplift, treatment_test, outcome_test, rate=0.4)
        score20 = uplift_score(uplift, treatment_test, outcome_test, rate=0.2)

    return score40, score20, uplift

"""