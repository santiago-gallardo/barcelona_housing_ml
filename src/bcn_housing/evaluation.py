from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate


def metrics(y_true, y_pred) -> dict:
    """Compute standard regression metrics on the original target scale."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def cv_metrics(
    estimator,
    X,
    y,
    *,
    seed: int = 42,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> dict:
    """Cross-validated metrics (mean/std) for a single estimator.

    Notes
    -----
    - Uses KFold with shuffling for reproducibility.
    - RMSE/MAE are computed via sklearn's negative scorers and converted back
      to positive values.
    - Intended for *robustness checks* on the training set.

    Parameters
    ----------
    estimator : sklearn estimator
        Any scikit-learn regressor, including wrappers like TransformedTargetRegressor.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    seed : int
        Random seed used for fold shuffling.
    n_splits : int
        Number of folds.
    n_jobs : int
        Parallel jobs used by cross_validate.

    Returns
    -------
    dict
        Keys: RMSE_mean, RMSE_std, MAE_mean, MAE_std, R2_mean, R2_std, folds
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    out = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        error_score="raise",
    )

    rmse_scores = -np.asarray(out["test_rmse"], dtype=float)
    mae_scores = -np.asarray(out["test_mae"], dtype=float)
    r2_scores = np.asarray(out["test_r2"], dtype=float)

    return {
        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std": float(np.std(rmse_scores, ddof=0)),
        "MAE_mean": float(np.mean(mae_scores)),
        "MAE_std": float(np.std(mae_scores, ddof=0)),
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores, ddof=0)),
        "folds": int(n_splits),
    }


def cv_summary_df(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a tidy DataFrame for CSV export."""
    if not rows:
        return pd.DataFrame(
            columns=[
                "Model",
                "n_features",
                "rows_used_train",
                "folds",
                "RMSE_mean",
                "RMSE_std",
                "MAE_mean",
                "MAE_std",
                "R2_mean",
                "R2_std",
            ]
        )
    df = pd.DataFrame(rows)
    # nice ordering
    cols = [
        "Model",
        "n_features",
        "rows_used_train",
        "folds",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "R2_mean",
        "R2_std",
    ]
    keep = [c for c in cols if c in df.columns]
    return df[keep].sort_values("RMSE_mean", ascending=True)
