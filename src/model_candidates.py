from __future__ import annotations

from typing import Dict, List, Tuple

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_candidates() -> Dict[str, Tuple[object, Dict[str, List]]]:
    """Return candidate classifiers and small grids for optional tuning.

    Note: Downstream pooled training may ignore the grids for speed and
    fit candidates with their defaults; grids are kept for extensibility.
    """
    return {
        "logreg": (
            LogisticRegression(
                max_iter=2000, solver="saga", multi_class="multinomial", n_jobs=-1
            ),
            {"C": [0.2, 0.5, 1.0, 2.0], "penalty": ["l2"]},
        ),
        "rf": (
            RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight="balanced_subsample"
            ),
            {
                "n_estimators": [200, 400],
                "min_samples_leaf": [1, 2],
                "max_depth": [None, 12],
            },
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
