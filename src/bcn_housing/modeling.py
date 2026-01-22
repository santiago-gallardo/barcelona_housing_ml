from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor

from sklearn.dummy import DummyRegressor


def make_dummy(strategy="median"):
    return DummyRegressor(strategy=strategy)


def make_lr() -> LinearRegression:
    return LinearRegression()


def make_tree(seed: int, max_depth: int = 2) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(max_depth=max_depth, random_state=seed)


def make_rf(
    seed: int,
    n_estimators: int = 100,
    min_samples_split: int = 3,
    max_depth=None,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )


def with_log_target(regressor):
    """
    Train on log1p(y), predict back on original scale (expm1).
    This keeps evaluation in the original PRICE scale.
    """
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def unwrap_regressor(model):
    """If model is TransformedTargetRegressor, return the inner regressor."""
    return getattr(model, "regressor_", model)
