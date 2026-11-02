"""
train_sota.py — SOTA Uplift Training Module
DragonNet (shared representation), Ensemble meta-learner,
Optuna hyperparameter optimization, nested cross-fitting.

Run:
    uv run --with "pandas,numpy,lightgbm,scikit-learn,mlxtend,econml,optuna,torch,torchvision" \\
        python3 src/train_sota.py --dataset x5 --model all
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from business_config import REGISTRY_PATH
from evaluate import UpliftEvaluator
from layer4_uplift_margin import XLearnerModel, _train_xlearner

# ---------------------------------------------------------------------------
# Optional torch import — DragonNet requires it
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def compute_auuc_from_arrays(uplift: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
    """Compute AUUC from raw arrays via UpliftEvaluator."""
    return UpliftEvaluator(n_bootstrap=0).evaluate(uplift, T, Y).auuc


# ---------------------------------------------------------------------------
# DragonNet (Shi et al. 2019)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class DragonNet(nn.Module):
        """
        Shi et al. 2019 — shared representation network for uplift.
        Architecture:
          - Shared: 3 FC layers (input -> 200 -> 200 -> 200) with ELU, BatchNorm, Dropout(0.2)
          - Treatment head: 200 -> 1 sigmoid (propensity)
          - Outcome heads (2): each 200 -> 100 -> 1 sigmoid (Y0, Y1)
          - Epsilon head: single scalar for targeted regularization
        Loss: BCE(propensity) + MSE(Y0,Y1) + targeted_reg * epsilon_loss
        """

        def __init__(self, input_dim: int, hidden_dim: int = 200,
                     dropout: float = 0.2):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            )

            # Treatment (propensity) head
            self.prop_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

            # Outcome heads Y0 and Y1
            self.y0_head = nn.Sequential(
                nn.Linear(hidden_dim, 100),
                nn.ELU(),
                nn.Linear(100, 1),
                nn.Sigmoid(),
            )
            self.y1_head = nn.Sequential(
                nn.Linear(hidden_dim, 100),
                nn.ELU(),
                nn.Linear(100, 1),
                nn.Sigmoid(),
            )

            # Epsilon scalar for targeted regularization
            self.epsilon = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            phi = self.shared(x)
            prop = self.prop_head(phi).squeeze(-1)          # (batch,)
            y0   = self.y0_head(phi).squeeze(-1)            # (batch,)
            y1   = self.y1_head(phi).squeeze(-1)            # (batch,)
            return prop, y0, y1, self.epsilon.expand(x.size(0))

        def dragonnet_loss(self, prop, y0, y1, epsilon,
                           T_batch, Y_batch, targeted_reg: float = 1.0):
            """
            Composite DragonNet loss:
              L_prop  = BCE(prop, T)
              L_y     = MSE(y_hat, Y) where y_hat = T*y1 + (1-T)*y0
              L_eps   = targeted regularization (TMLE-inspired correction)
            """
            bce = nn.BCELoss()
            mse = nn.MSELoss()

            # Propensity loss
            L_prop = bce(prop, T_batch)

            # Outcome loss — supervised on observed Y using the correct head
            y_hat = T_batch * y1 + (1.0 - T_batch) * y0
            L_y = mse(y_hat, Y_batch)

            # Targeted regularization (TMLE-style correction term)
            # Encourages epsilon to correct for residual bias in outcome models
            h = T_batch / (prop.detach() + 1e-7) - (1.0 - T_batch) / (1.0 - prop.detach() + 1e-7)
            L_eps = torch.mean((Y_batch - y_hat.detach() - epsilon * h) ** 2)

            return L_prop + L_y + targeted_reg * L_eps


class DragonNetWrapper:
    """
    Sklearn-compatible wrapper around DragonNet.
    Exposes predict_uplift(X) -> np.ndarray.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 200, dropout: float = 0.2,
                 lr: float = 1e-3, epochs: int = 100, batch_size: int = 512,
                 targeted_reg: float = 1.0, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("DragonNet requires PyTorch.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.targeted_reg = targeted_reg
        self.device = torch.device(device)
        self.model = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "DragonNetWrapper":
        self.model = DragonNet(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(X_t, T_t, Y_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_b, T_b, Y_b in loader:
                optimizer.zero_grad()
                prop, y0, y1, eps = self.model(X_b)
                loss = self.model.dragonnet_loss(
                    prop, y0, y1, eps, T_b, Y_b, self.targeted_reg
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"    [DragonNet] epoch {epoch+1}/{self.epochs}  "
                      f"loss={epoch_loss/len(loader):.4f}")

        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Returns CATE estimate: E[Y1 - Y0 | X]."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict_uplift().")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            _, y0, y1, _ = self.model(X_t)
            cate = (y1 - y0).cpu().numpy()
        return cate


# ---------------------------------------------------------------------------
# X-Learner wrapper (thin adaptor over layer4_uplift_margin._train_xlearner)
# ---------------------------------------------------------------------------

class XLearnerWrapper:
    """
    Thin adaptor: shapes numpy arrays into DataFrames and delegates
    to layer4_uplift_margin._train_xlearner, then wraps the returned
    XLearnerModel.  Exposes predict_uplift(X) -> np.ndarray.
    """

    def __init__(self, n_estimators: int = 200):
        self.n_estimators = n_estimators
        self._model: XLearnerModel | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "XLearnerWrapper":
        feat_cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feat_cols)
        df["was_treated"] = T.astype(int)
        df["converted"]   = Y.astype(int)
        treat_df = df[df["was_treated"] == 1].copy()
        ctrl_df  = df[df["was_treated"] == 0].copy()
        self._model = _train_xlearner(treat_df, ctrl_df, feat_cols,
                                      n_estimators=self.n_estimators)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_uplift(X)


# ---------------------------------------------------------------------------
# R-Learner wrapper (CausalForestDML)
# ---------------------------------------------------------------------------

class RLearnerWrapper:
    """
    Thin adaptor over EconML CausalForestDML exposing predict_uplift().
    Accepts optional hyperparams dict (from tune_rlearner).
    """

    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 10,
                 max_depth: int = None):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self._model = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "RLearnerWrapper":
        from econml.dml import CausalForestDML

        kwargs = dict(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            discrete_treatment=True,
            random_state=42,
            cv=5,
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4),
            model_t=GradientBoostingClassifier(n_estimators=100, max_depth=4),
        )
        if self.max_depth is not None:
            kwargs["max_depth"] = self.max_depth

        self._model = CausalForestDML(**kwargs)
        self._model.fit(Y, T, X=X)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self._model.effect(X).ravel()


# ---------------------------------------------------------------------------
# Optuna tuning for CausalForestDML
# ---------------------------------------------------------------------------

def tune_rlearner(X_tr: np.ndarray, T_tr: np.ndarray, Y_tr: np.ndarray,
                  n_trials: int = 20) -> dict:
    """
    Optuna tuning for CausalForestDML hyperparameters.
    Returns best hyperparams dict with keys: n_estimators, min_samples_leaf, max_depth.
    Objective: maximize AUUC on a 20% validation fold.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [tune_rlearner] optuna not available — returning defaults.")
        return {"n_estimators": 200, "min_samples_leaf": 10, "max_depth": None}

    X_fit, X_val, T_fit, T_val, Y_fit, Y_val = train_test_split(
        X_tr, T_tr, Y_tr, test_size=0.2, random_state=42, stratify=T_tr
    )

    def objective(trial):
        n_est      = trial.suggest_int("n_estimators",     50,  400, step=50)
        min_leaf   = trial.suggest_int("min_samples_leaf",  5,   50)
        max_depth  = trial.suggest_categorical("max_depth", [None, 3, 4, 5, 6])

        try:
            model = RLearnerWrapper(
                n_estimators=n_est,
                min_samples_leaf=min_leaf,
                max_depth=max_depth,
            )
            model.fit(X_fit, T_fit, Y_fit)
            uplift_val = model.predict_uplift(X_val)
            score = compute_auuc_from_arrays(uplift_val, T_val, Y_val)
        except Exception:
            score = float("-inf")
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  [tune_rlearner] best params (AUUC={study.best_value:.6f}): {best}")
    return {
        "n_estimators":    best["n_estimators"],
        "min_samples_leaf": best["min_samples_leaf"],
        "max_depth":        best["max_depth"],
    }


# ---------------------------------------------------------------------------
# Ensemble uplift model
# ---------------------------------------------------------------------------

class EnsembleUpliftModel:
    """
    Weighted ensemble of X-Learner, R-Learner, and DragonNet.

    Weights are determined by out-of-fold (OOF) AUUC on training data —
    higher AUUC -> higher weight, softmax-normalized so weights sum to 1.

    If DragonNet is unavailable (torch missing), the ensemble uses
    X-Learner + R-Learner only.
    """

    def __init__(self, xlearner_params: dict = None, rlearner_params: dict = None,
                 dragonnet_params: dict = None, n_folds: int = 3):
        self.xlearner_params  = xlearner_params  or {}
        self.rlearner_params  = rlearner_params  or {}
        self.dragonnet_params = dragonnet_params or {}
        self.n_folds          = n_folds
        self._models          = {}
        self._weights         = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_xlearner(self) -> XLearnerWrapper:
        return XLearnerWrapper(**self.xlearner_params)

    def _make_rlearner(self) -> RLearnerWrapper:
        return RLearnerWrapper(**self.rlearner_params)

    def _make_dragonnet(self, input_dim: int):
        if not TORCH_AVAILABLE:
            return None
        params = dict(
            input_dim=input_dim,
            hidden_dim=self.dragonnet_params.get("hidden_dim", 200),
            dropout=self.dragonnet_params.get("dropout", 0.2),
            lr=self.dragonnet_params.get("lr", 1e-3),
            epochs=self.dragonnet_params.get("epochs", 100),
            batch_size=self.dragonnet_params.get("batch_size", 512),
            targeted_reg=self.dragonnet_params.get("targeted_reg", 1.0),
        )
        return DragonNetWrapper(**params)

    def _oof_auuc(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                  model_name: str) -> float:
        """
        Estimate AUUC via K-fold OOF predictions for a given model type.
        Returns average AUUC across folds.
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        auucs = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_vl = X[train_idx], X[val_idx]
            T_tr, T_vl = T[train_idx], T[val_idx]
            Y_tr, Y_vl = Y[train_idx], Y[val_idx]

            # Need at least some treated and control in both splits
            if T_tr.sum() < 5 or (1 - T_tr).sum() < 5:
                continue
            if T_vl.sum() < 2 or (1 - T_vl).sum() < 2:
                continue

            try:
                if model_name == "xlearner":
                    m = self._make_xlearner()
                    m.fit(X_tr, T_tr, Y_tr)
                    uplift = m.predict_uplift(X_vl)
                elif model_name == "rlearner":
                    m = self._make_rlearner()
                    m.fit(X_tr, T_tr, Y_tr)
                    uplift = m.predict_uplift(X_vl)
                elif model_name == "dragonnet":
                    m = self._make_dragonnet(X_tr.shape[1])
                    if m is None:
                        continue
                    m.fit(X_tr, T_tr, Y_tr)
                    uplift = m.predict_uplift(X_vl)
                else:
                    continue

                score = compute_auuc_from_arrays(uplift, T_vl, Y_vl)
                auucs.append(score)
                print(f"    [OOF] {model_name} fold {fold_idx+1}/{self.n_folds}  "
                      f"AUUC={score:.6f}")
            except Exception as e:
                print(f"    [OOF] {model_name} fold {fold_idx+1} failed: {e}")

        return float(np.mean(auucs)) if auucs else 0.0

    @staticmethod
    def _softmax_weights(scores: dict) -> dict:
        """Softmax-normalize raw AUUC scores to ensemble weights."""
        keys = list(scores.keys())
        vals = np.array([scores[k] for k in keys], dtype=float)
        # Shift so minimum is 0 before softmax (handles negative AUUC gracefully)
        vals = vals - vals.min()
        exp_vals = np.exp(vals)
        norm = exp_vals / exp_vals.sum()
        return {k: float(w) for k, w in zip(keys, norm)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "EnsembleUpliftModel":
        """
        1. Compute OOF AUUC for each base model.
        2. Derive softmax weights.
        3. Refit each base model on the full training data.
        """
        print("\n[EnsembleUpliftModel] Computing out-of-fold AUUC weights...")

        model_names = ["xlearner", "rlearner"]
        if TORCH_AVAILABLE:
            model_names.append("dragonnet")
        else:
            print("  DragonNet skipped (torch not available) — ensemble uses 2 models.")

        oof_scores = {}
        for name in model_names:
            print(f"\n  >> OOF evaluation: {name}")
            oof_scores[name] = self._oof_auuc(X, T, Y, name)
            print(f"     mean OOF AUUC = {oof_scores[name]:.6f}")

        self._weights = self._softmax_weights(oof_scores)
        print(f"\n  Ensemble weights (softmax of OOF AUUC):")
        for name, w in self._weights.items():
            print(f"    {name:<12}  {w:.4f}")

        # Refit on full data
        print("\n[EnsembleUpliftModel] Refitting base models on full training data...")

        xl = self._make_xlearner()
        xl.fit(X, T, Y)
        self._models["xlearner"] = xl
        print("  X-Learner: done")

        rl = self._make_rlearner()
        rl.fit(X, T, Y)
        self._models["rlearner"] = rl
        print("  R-Learner: done")

        if TORCH_AVAILABLE:
            dn = self._make_dragonnet(X.shape[1])
            dn.fit(X, T, Y)
            self._models["dragonnet"] = dn
            print("  DragonNet: done")

        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Weighted average CATE across all base models."""
        if not self._models:
            raise RuntimeError("Call fit() before predict_uplift().")

        total = np.zeros(len(X), dtype=float)
        for name, model in self._models.items():
            w = self._weights.get(name, 0.0)
            total += w * model.predict_uplift(X)
        return total

    def get_weights(self) -> dict:
        """Returns {'xlearner': w1, 'rlearner': w2, 'dragonnet': w3}."""
        return dict(self._weights)


# ---------------------------------------------------------------------------
# Dataset loaders (mirrors train_rlearner.py)
# ---------------------------------------------------------------------------

def load_hillstrom():
    path = "data/hillstrom.csv"
    df = pd.read_csv(path)

    if "segment" in df.columns:
        treatment_raw_col = "segment"
        outcome_col = "visit"
    else:
        treatment_raw_col = "treatment"
        outcome_col = "target"

    df["treatment_bin"] = (df[treatment_raw_col] != "No E-Mail").astype(int)

    cat_cols = [c for c in ["history_segment", "zip_code", "channel"] if c in df.columns]
    num_cols = [c for c in ["recency", "history", "mens", "womens", "newbie"] if c in df.columns]

    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    feature_cols = num_cols + [c + "_enc" for c in cat_cols]
    X = df[feature_cols].astype(float).values
    T = df["treatment_bin"].values
    Y = df[outcome_col].values
    return X, T, Y, feature_cols


def load_x5():
    path = "data/x5_retailhero/uplift_train.csv.gz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"X5 dataset not found at {path}")
    df = pd.read_csv(path)
    treatment_col = "treatment_flg" if "treatment_flg" in df.columns else "treatment"
    feature_cols = [c for c in df.columns if c not in ["client_id", treatment_col, "target"]]
    X = df[feature_cols].fillna(0).astype(float).values
    T = df[treatment_col].values
    Y = df["target"].values
    return X, T, Y, feature_cols


def load_pos():
    path = "data/raw_pos_transactions.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"POS dataset not found at {path}")
    from feature_engineering import extract_features
    raw = pd.read_csv(path)
    features_df = extract_features(raw)
    treatment_col = "is_treated"
    outcome_col   = "post_promo_visit"
    required = [treatment_col, outcome_col, "customer_id"]
    for col in required:
        if col not in raw.columns and col not in features_df.columns:
            raise ValueError(f"Column '{col}' not found in POS data.")
    label_cols = [c for c in required if c in raw.columns]
    if label_cols:
        labels = raw[label_cols].drop_duplicates("customer_id")
        features_df["customer_id"] = features_df["customer_id"].astype(str)
        labels["customer_id"] = labels["customer_id"].astype(str)
        features_df = features_df.merge(labels, on="customer_id", how="inner")
    id_cols = {"customer_id", treatment_col, outcome_col}
    feature_cols = [c for c in features_df.columns if c not in id_cols]
    X = features_df[feature_cols].fillna(0).astype(float).values
    T = features_df[treatment_col].values
    Y = features_df[outcome_col].values
    return X, T, Y, feature_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SOTA Uplift Training: DragonNet + Ensemble + Optuna"
    )
    parser.add_argument(
        "--dataset", choices=["hillstrom", "x5", "pos"], default="hillstrom",
        help="Dataset to train on (default: hillstrom)",
    )
    parser.add_argument(
        "--model",
        choices=["dragonnet", "ensemble", "tune_rlearner", "all"],
        default="all",
        help="Which component to run (default: all)",
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=20,
        help="Number of Optuna trials for tune_rlearner (default: 20)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="DragonNet training epochs (default: 100)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  train_sota.py  |  dataset={args.dataset}  model={args.model}")
    print(f"  torch available: {TORCH_AVAILABLE}")
    print(f"{'='*60}")

    # --- Load data ---
    print(f"\n[1] Loading {args.dataset} dataset...")
    if args.dataset == "hillstrom":
        X, T, Y, feature_cols = load_hillstrom()
    elif args.dataset == "x5":
        X, T, Y, feature_cols = load_x5()
    else:
        X, T, Y, feature_cols = load_pos()

    print(f"    samples={len(Y):,}  features={len(feature_cols)}  "
          f"treated={T.sum():,}  control={(1-T).sum():,}")

    # Train/test split
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42, stratify=T
    )
    print(f"    train={len(Y_train):,}  test={len(Y_test):,}")

    results = {}

    # --- DragonNet ---
    if args.model in ("dragonnet", "all"):
        print(f"\n[DragonNet] Training...")
        if not TORCH_AVAILABLE:
            print("  DragonNet skipped (torch not available)")
        else:
            dn = DragonNetWrapper(
                input_dim=X_train.shape[1],
                epochs=args.epochs,
            )
            dn.fit(X_train, T_train, Y_train)
            dn_uplift = dn.predict_uplift(X_test)
            dn_auuc = compute_auuc_from_arrays(dn_uplift, T_test, Y_test)
            print(f"  DragonNet AUUC = {dn_auuc:.6f}")
            results["dragonnet"] = dn_auuc

            save_path = os.path.join(REGISTRY_PATH, f"dragonnet_{args.dataset}.pkl")
            os.makedirs(REGISTRY_PATH, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(dn, f)
            print(f"  Saved to: {save_path}")

    # --- Optuna tuning for R-Learner ---
    best_rlearner_params = {"n_estimators": 200, "min_samples_leaf": 10, "max_depth": None}
    if args.model in ("tune_rlearner", "all"):
        print(f"\n[Optuna] Tuning R-Learner (CausalForestDML), "
              f"n_trials={args.optuna_trials}...")
        best_rlearner_params = tune_rlearner(
            X_train, T_train, Y_train, n_trials=args.optuna_trials
        )

    # --- Ensemble ---
    if args.model in ("ensemble", "all"):
        print(f"\n[Ensemble] Training EnsembleUpliftModel...")
        ensemble = EnsembleUpliftModel(
            rlearner_params={
                "n_estimators":    best_rlearner_params["n_estimators"],
                "min_samples_leaf": best_rlearner_params["min_samples_leaf"],
                "max_depth":        best_rlearner_params["max_depth"],
            },
            dragonnet_params={"epochs": args.epochs},
        )
        ensemble.fit(X_train, T_train, Y_train)
        ens_uplift = ensemble.predict_uplift(X_test)
        ens_auuc = compute_auuc_from_arrays(ens_uplift, T_test, Y_test)
        results["ensemble"] = ens_auuc

        print(f"\n  Ensemble weights: {ensemble.get_weights()}")
        print(f"  Ensemble AUUC   = {ens_auuc:.6f}")

        save_path = os.path.join(REGISTRY_PATH, f"ensemble_sota_{args.dataset}.pkl")
        os.makedirs(REGISTRY_PATH, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(ensemble, f)
        print(f"  Saved to: {save_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  AUUC Summary — dataset={args.dataset}")
    print(f"{'='*60}")
    for model_name, auuc in results.items():
        print(f"  {model_name:<20}  {auuc:>10.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
