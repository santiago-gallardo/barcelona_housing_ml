from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from bcn_housing.config import (
    SEED,
    TEST_SIZE,
    TARGET,
    DEFAULT_DATA_PATH,
    FEATURES_LR_BENCH,
    FEATURES_TREE_2,
    FEATURES_RF10,
    FEATURES_RF5,
    FINAL_MODEL_NAME,
    MODEL_LABELS,
    MODEL_ORDER,
)
from bcn_housing.data import load_csv, basic_checks, prepare_xy
from bcn_housing.modeling import make_lr, make_tree, make_rf, with_log_target, make_dummy
from bcn_housing.evaluation import metrics, cv_metrics, cv_summary_df
from bcn_housing.plots import (
    save_feature_importance,
    save_ytrue_vs_ypred,
    save_residuals_hist_log,
    save_metric_evolution,
)


def extract_params_summary(model) -> dict:
    """
    Keep the JSON clean: store only the most relevant hyperparameters.
    Supports TransformedTargetRegressor wrappers (log(y)).
    """
    out = {}

    # unwrap TransformedTargetRegressor if present
    if hasattr(model, "regressor_"):
        out["wrapper"] = "TransformedTargetRegressor"
        out["target_transform"] = "log1p/expm1"
        reg = model.regressor_
    else:
        reg = model

    est = reg.__class__.__name__
    params = reg.get_params()

    if est == "DummyRegressor":
        keep = ["strategy", "constant", "quantile"]
    elif est == "LinearRegression":
        keep = ["fit_intercept", "positive", "n_jobs"]
    elif est == "DecisionTreeRegressor":
        keep = ["max_depth", "min_samples_split",
                "min_samples_leaf", "random_state"]
    elif est == "RandomForestRegressor":
        keep = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "random_state",
            "n_jobs",
        ]
    else:
        # fallback (small set)
        keep = ["random_state"]

    out["estimator"] = est
    out["params"] = {k: params.get(k) for k in keep if k in params}
    return out


def fit_evaluate_model(df: pd.DataFrame, features: list[str], model):
    """
    Prepare X/y, split, fit, predict, return metrics and diagnostics.
    """
    X, y, prep_report = prepare_xy(df, features, TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m = metrics(y_test, y_pred)
    split_info = {"n_train": int(len(y_train)), "n_test": int(len(y_test))}
    return m, prep_report, split_info, (y_test, y_pred), model, (X_train, y_train)


def main():
    parser = argparse.ArgumentParser(
        description="Train Barcelona Housing ML models.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV dataset (default: data/raw/Barcelona_2018.csv)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Reports output folder (default: reports/)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts output folder (default: artifacts/)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run k-fold CV on the TRAIN split for robustness and save reports/cv_summary.csv",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for CV (default: 5)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    reports_dir = Path(args.reports_dir)
    artifacts_dir = Path(args.artifacts_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(str(data_path))

    # Candidate models aligned to your story
    candidates = [
        ("Benchmark Dummy (median)", FEATURES_LR_BENCH, make_dummy("median")),
        ("LR (area)", FEATURES_LR_BENCH, make_lr()),
        ("DecisionTree (2 vars, depth=2)",
         FEATURES_TREE_2, make_tree(SEED, max_depth=2)),
        ("RandomForest (10 vars)", FEATURES_RF10, make_rf(SEED)),
        ("RandomForest (5 vars) + log(y)",
         FEATURES_RF5, with_log_target(make_rf(SEED))),
    ]

    rows = []
    cv_rows = []
    trained_cache = {}

    # Ensure these exist even when --cv is NOT used
    cv_df = None
    cv_summary_path = None

    for name, feats, model in candidates:
        # clone BEFORE fitting so we can use it safely for cross_validate()
        model_for_cv = clone(model)

        m, prep_report, split_info, (y_test, y_pred), fitted, (X_train, y_train) = fit_evaluate_model(
            df, feats, model
        )

        row = {
            "Model": name,
            "n_features": int(len(feats)),
            "RMSE": float(m["RMSE"]),
            "MAE": float(m["MAE"]),
            "R2": float(m["R2"]),
            "rows_in": int(prep_report.rows_in),
            "rows_used": int(prep_report.rows_out),
            "dropped_missing": int(prep_report.dropped_missing),
            "n_train": int(split_info["n_train"]),
            "n_test": int(split_info["n_test"]),
        }
        rows.append(row)

        trained_cache[name] = {
            "model": fitted,
            "features": feats,
            "y_test": y_test,
            "y_pred": y_pred,
            "split": split_info,
        }

        # Optional: CV on TRAIN split only (robustness check)
        if args.cv:
            cvm = cv_metrics(
                model_for_cv,
                X_train,
                y_train,
                seed=SEED,
                n_splits=args.cv_folds,
            )
            cv_rows.append(
                {
                    "Model": name,
                    "n_features": int(len(feats)),
                    "rows_used_train": int(len(y_train)),
                    **cvm,
                }
            )

    results_df = pd.DataFrame(rows).sort_values("RMSE", ascending=True)
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    if args.cv:
        cv_df = cv_summary_df(cv_rows)
        cv_summary_path = reports_dir / "cv_summary.csv"
        cv_df.to_csv(cv_summary_path, index=False)

    # Final model is your chosen one (parsimonious), not necessarily best RMSE
    best_name = FINAL_MODEL_NAME
    if best_name not in trained_cache:
        raise ValueError(
            f"FINAL_MODEL_NAME='{best_name}' not found in trained_cache. "
            "Check config.py and candidate names."
        )

    best_obj = trained_cache[best_name]["model"]
    best_feats = trained_cache[best_name]["features"]
    y_test = trained_cache[best_name]["y_test"]
    y_pred = trained_cache[best_name]["y_pred"]

    best_by_rmse = results_df.iloc[0]["Model"]

    # Rows for JSON
    final_row = results_df[results_df["Model"] == best_name].iloc[0].to_dict()
    best_by_rmse_row = results_df.iloc[0].to_dict()

    # Trade-off final vs best (HOLDOUT)
    delta_final_vs_best = {
        "rmse_abs": float(final_row["RMSE"] - best_by_rmse_row["RMSE"]),
        "rmse_pct": float((final_row["RMSE"] / best_by_rmse_row["RMSE"]) - 1.0),
        "r2_abs": float(final_row["R2"] - best_by_rmse_row["R2"]),
    }

    # Save final model artifact
    dump(best_obj, artifacts_dir / "model.joblib")

    # Save feature list used by the final model (for batch prediction / Streamlit)
    features_out = artifacts_dir / "features.json"
    with open(features_out, "w", encoding="utf-8") as f:
        json.dump(
            {"model_name": best_name, "features": list(best_feats)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Figures (final model)
    save_ytrue_vs_ypred(y_test, y_pred, reports_dir /
                        "figures" / "y_true_vs_pred_final.png")
    save_residuals_hist_log(
        y_true=y_test,
        y_pred=y_pred,
        out_path=reports_dir / "figures" / "residuals_hist_final.png",
    )

    # Feature importance: justification (RF10) + final
    rf10_name = "RandomForest (10 vars)"
    if rf10_name in trained_cache:
        save_feature_importance(
            trained_cache[rf10_name]["model"],
            trained_cache[rf10_name]["features"],
            reports_dir / "figures" / "feature_importance_rf10_selection.png",
            title="Feature Importance (Feature Selection Step)",
            subtitle="Random Forest (10 features)",
        )

    if "RandomForest" in best_name:
        save_feature_importance(
            best_obj,
            best_feats,
            reports_dir / "figures" / "feature_importance_rf5_final.png",
            title="Feature Importance (Final Model)",
            subtitle="Random Forest (5 features) + log(y)",
        )

    # Comparison plots across models
    save_metric_evolution(
        results_df=results_df,
        model_order=MODEL_ORDER,
        model_labels=MODEL_LABELS,
        metric_col="RMSE",
        out_path=reports_dir / "figures" / "rmse_evolution.png",
        title="RMSE Evolution by Model",
        ylabel="RMSE (€)",
    )

    # IMPORTANT: allow negative R2 so Dummy appears
    save_metric_evolution(
        results_df=results_df,
        model_order=MODEL_ORDER,
        model_labels=MODEL_LABELS,
        metric_col="R2",
        out_path=reports_dir / "figures" / "r2_evolution.png",
        title="R² Evolution by Model",
        ylabel="R²",
        ylim=(-0.2, 1),
    )

    # Final model details
    final_reg = getattr(best_obj, "regressor_", best_obj)  # unwrap if TTR
    final_model_details = {
        "model_name": best_name,
        "features": list(best_feats),
        "n_features": int(len(best_feats)),
        "estimator_class": final_reg.__class__.__name__,
        "params_summary": extract_params_summary(best_obj),
    }

    # Error stats (final)
    abs_err = np.abs(np.asarray(y_test, dtype=float) -
                     np.asarray(y_pred, dtype=float))
    final_error_stats = {
        "abs_error_mean": float(abs_err.mean()),
        "abs_error_median": float(np.median(abs_err)),
        "abs_error_p90": float(np.quantile(abs_err, 0.90)),
        "abs_error_p95": float(np.quantile(abs_err, 0.95)),
    }

    # Run metadata
    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }

    # Include all model rows in JSON too
    models_all = results_df.to_dict(orient="records")

    # ---- CV rows for metrics.json (so everything is visible in one place) ----
    cv_summary_rows = None
    cv_best_by_rmse_mean_row = None
    cv_final_vs_cv_best = None

    if args.cv and cv_df is not None and not cv_df.empty:
        cv_df_json = cv_df.copy()

        # Make JSON readable
        for col in ["RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"]:
            if col in cv_df_json.columns:
                cv_df_json[col] = cv_df_json[col].astype(float).round(3)

        # Ensure sorted by RMSE_mean
        if "RMSE_mean" in cv_df_json.columns:
            cv_df_json = cv_df_json.sort_values("RMSE_mean", ascending=True)

        cv_summary_rows = cv_df_json.to_dict(orient="records")
        cv_best_by_rmse_mean_row = cv_summary_rows[0] if cv_summary_rows else None

        # Optional: how far the selected final model is from the CV best
        try:
            final_rmse_cv = float(
                cv_df.loc[cv_df["Model"] == best_name, "RMSE_mean"].iloc[0]
            )
            best_rmse_cv = float(cv_df["RMSE_mean"].min())
            cv_final_vs_cv_best = {
                "final_rmse_mean": round(final_rmse_cv, 3),
                "best_rmse_mean": round(best_rmse_cv, 3),
                "delta_rmse_mean": round(final_rmse_cv - best_rmse_cv, 3),
                "delta_pct": round((final_rmse_cv - best_rmse_cv) / best_rmse_cv * 100, 2),
            }
        except Exception:
            cv_final_vs_cv_best = None
    # -------------------------------------------------------------------------

    payload = {
        "data_path": str(data_path),
        "basic_checks": basic_checks(df, TARGET),

        "run_metadata": run_metadata,
        "split": {
            "seed": int(SEED),
            "test_size": float(TEST_SIZE),
            "final_n_train": int(trained_cache[best_name]["split"]["n_train"]),
            "final_n_test": int(trained_cache[best_name]["split"]["n_test"]),
        },

        "cv": {
            "enabled": bool(args.cv),
            "folds": int(args.cv_folds) if args.cv else None,
            "note": "CV is run on the TRAIN split only (robustness check)." if args.cv else None,
            "summary_rows": cv_summary_rows,
            "best_by_rmse_mean_row": cv_best_by_rmse_mean_row,
            "final_vs_best": cv_final_vs_cv_best,
        },

        "best_by_rmse": best_by_rmse,
        "best_by_rmse_row": best_by_rmse_row,

        "final_model_selected": best_name,
        "final_row": final_row,
        "selection_reason": "parsimony (simpler model with comparable performance)",
        "delta_final_vs_best": delta_final_vs_best,

        "models": models_all,
        "final_model_details": final_model_details,
        "final_error_stats": final_error_stats,

        "reports": {
            "comparison_csv": str(reports_dir / "model_comparison.csv"),
            "figures_dir": str(reports_dir / "figures"),
            "cv_summary_csv": str(cv_summary_path) if cv_summary_path else None,
        },
        "artifacts": {"model_path": str(artifacts_dir / "model.joblib"), "features_path": str(features_out)},
    }

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ Training finished.")
    print(results_df)


if __name__ == "__main__":
    main()
