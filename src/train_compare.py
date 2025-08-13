from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .data import prepare_dataset, train_test_split_time, save_metadata
from .features import FeatureParams, LabelParams


ARTIFACTS_DIR = Path("artifacts/model")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ALL_MODELS_DIR = Path("artifacts/models")
ALL_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_candidates() -> Dict[str, Tuple[object, Dict[str, List]]]:
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, solver="saga", multi_class="multinomial", n_jobs=-1),
            {"C": [0.2, 0.5, 1.0, 2.0], "penalty": ["l2"]},
        ),
        "rf": (
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
            {"n_estimators": [200, 400], "min_samples_leaf": [1, 2], "max_depth": [None, 12]},
        ),
        "et": (
            ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            {"n_estimators": [400, 800], "min_samples_leaf": [1, 2]},
        ),
        "gb": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        ),
        "hgbt": (
            HistGradientBoostingClassifier(random_state=42),
            {"max_depth": [None, 8], "learning_rate": [0.05, 0.1]},
        ),
        "mlp": (
            MLPClassifier(random_state=42, max_iter=500),
            {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-4, 1e-3]},
        ),
        "svc": (
            SVC(probability=True, random_state=42),
            {"C": [0.5, 1.0, 2.0], "kernel": ["rbf"]},
        ),
    }


def evaluate_cv(model, params_grid: Dict[str, List], X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> Tuple[float, Dict[str, object]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = -np.inf
    best_params: Dict[str, object] = {}
    # simple grid search (small grids)
    from itertools import product

    keys = list(params_grid.keys())
    values = [params_grid[k] for k in keys]
    for combo in product(*values):
        params = dict(zip(keys, combo))
        scores: List[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            clf = model.__class__(**{**model.get_params(), **params})
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro")
            scores.append(score)
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_score, best_params


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("market-master-actions")

    feature_params = FeatureParams()
    label_params = LabelParams()
    ds, feature_cols = prepare_dataset("data/raw/sample_ohlcv.csv", feature_params, label_params)
    X_train, y_train, X_test, y_test = train_test_split_time(ds, feature_cols, test_ratio=0.2)

    candidates = get_candidates()

    summary: List[Dict[str, object]] = []
    with mlflow.start_run(run_name="model_comparison"):
        for name, (base_model, grid) in candidates.items():
            with mlflow.start_run(run_name=f"cv_{name}", nested=True):
                mlflow.log_param("candidate", name)
                cv_score, best_params = evaluate_cv(base_model, grid, X_train, y_train, n_splits=3)
                mlflow.log_metrics({"cv_f1_macro": cv_score})
                mlflow.log_params({f"cv_{k}": v for k, v in best_params.items()})

                # train with best params on full train
                model = base_model.__class__(**{**base_model.get_params(), **best_params})
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                f1m = f1_score(y_test, y_pred, average="macro")
                acc = accuracy_score(y_test, y_pred)
                mlflow.log_metrics({"test_f1_macro": f1m, "test_accuracy": acc})
                # keep a local snapshot for selection later
                tmp_path = ALL_MODELS_DIR / f"{name}.pkl"
                joblib.dump(model, tmp_path)
                summary.append({"name": name, "f1_macro": f1m, "acc": acc, "path": str(tmp_path), "params": best_params})

        # pick best by f1_macro, then accuracy
        summary.sort(key=lambda r: (r["f1_macro"], r["acc"]), reverse=True)
        best = summary[0]
        # move best to canonical model path
        best_src = Path(best["path"])  # type: ignore
        final_path = ARTIFACTS_DIR / "model.pkl"
        joblib.dump(joblib.load(best_src), final_path)
        mlflow.log_param("best_model", best["name"])  # type: ignore
        mlflow.log_metrics({"best_f1_macro": best["f1_macro"], "best_accuracy": best["acc"]})  # type: ignore

        meta_path = save_metadata(
            ARTIFACTS_DIR,
            feature_columns=feature_cols,
            feature_params=feature_params,
            label_params=label_params,
            dataset_info={"train_rows": str(len(X_train)), "test_rows": str(len(X_test)), "best_model": best["name"]},  # type: ignore
        )
        # Log artifacts: selected and per-model
        mlflow.log_artifact(final_path.as_posix(), artifact_path="selected")
        mlflow.log_artifact(meta_path.as_posix(), artifact_path="selected")
        # Write comparison summary
        cmp_json = ALL_MODELS_DIR / "summary.json"
        cmp_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(cmp_json.as_posix(), artifact_path="comparison")

        print(json.dumps({"best_model": best["name"], "f1_macro": best["f1_macro"], "accuracy": best["acc"]}, indent=2))  # type: ignore


if __name__ == "__main__":
    main()


