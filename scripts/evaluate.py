from pathlib import Path
import pandas as pd


def main():
    comp_path = Path("reports/model_comparison.csv")
    if not comp_path.exists():
        raise FileNotFoundError(
            "Missing reports/model_comparison.csv. Run scripts/train.py first."
        )

    comp_df = pd.read_csv(comp_path).sort_values("RMSE")
    print("\n=== Holdout comparison (train/test split) ===")
    print(comp_df)

    cv_path = Path("reports/cv_summary.csv")
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path).sort_values("RMSE_mean")
        print("\n=== Cross-validation (TRAIN split only) ===")
        print(cv_df)
    else:
        print("\n(No CV summary found. Run train.py with --cv to generate it.)")


if __name__ == "__main__":
    main()
