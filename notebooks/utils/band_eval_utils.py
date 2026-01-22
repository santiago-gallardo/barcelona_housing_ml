import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import r2_score

# ---------- helpers ----------


def euro_fmt(x, pos=None):
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"€{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"€{x/1_000:.0f}k"
    return f"€{x:.0f}"


def band_ranges_from_quantiles(y_true, q=4):
    """
    Banding by explicit quantile thresholds.
    q=4 => Q1:<q25, Q2:[q25,q50), Q3:[q50,q75), Q4:>=q75
    """
    y_true = pd.Series(y_true).astype(float)

    if q < 2:
        raise ValueError("q must be >= 2")

    probs = [i / q for i in range(1, q)]
    cuts = y_true.dropna().quantile(probs).values

    bands = pd.Series(pd.NA, index=y_true.index, dtype="object")
    bands[y_true < cuts[0]] = "Q1"

    for i in range(1, q - 1):
        lo = cuts[i - 1]
        hi = cuts[i]
        bands[(y_true >= lo) & (y_true < hi)] = f"Q{i+1}"

    bands[y_true >= cuts[-1]] = f"Q{q}"

    cat_order = [f"Q{i}" for i in range(1, q + 1)]
    bands = pd.Categorical(bands, categories=cat_order, ordered=True)

    tmp = pd.DataFrame({"y_true": y_true, "band": bands})
    ranges = (
        tmp.dropna(subset=["band"])
        .groupby("band", observed=True)["y_true"]
        .agg(["min", "max"])
        .round(0)
        .astype(int)
    )

    label_map = {}
    for b, row in ranges.iterrows():
        label_map[b] = f"{b} ({euro_fmt(row['min'])}–{euro_fmt(row['max'])})"

    return bands, label_map

# ---------- core analysis ----------


def price_band_error_analysis(y_test, y_pred, q=4):
    y_true = pd.Series(y_test).astype(float).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(float).reset_index(drop=True)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["err"] = df["y_pred"] - df["y_true"]
    df["abs_err"] = df["err"].abs()
    df["sq_err"] = df["err"] ** 2

    df["band"], label_map = band_ranges_from_quantiles(df["y_true"], q=q)
    df["band_label"] = df["band"].map(label_map)

    gb = df.groupby(["band", "band_label"], observed=True)

    summary = gb.agg(
        Count=("y_true", "count"),
        RMSE=("sq_err", lambda s: float(np.sqrt(s.mean()))),
        MAE=("abs_err", "mean"),
        Bias__pred_true=("err", "mean"),
        P90_abs_err=("abs_err", lambda s: float(np.quantile(s, 0.90))),
    ).reset_index()

    # ✅ sin warning (include_groups=False)
    r2_vals = (
        gb.apply(
            lambda g: r2_score(g["y_true"], g["y_pred"]
                               ) if len(g) >= 2 else np.nan,
            include_groups=False
        )
        .rename("R2")
        .reset_index()
    )

    summary = summary.merge(r2_vals, on=["band", "band_label"], how="left")
    summary = summary.sort_values("band").drop(columns=["band"])

    summary = summary.rename(columns={
        "Bias__pred_true": "Bias (pred-true)",
        "P90_abs_err": "P90 abs err"
    })

    return df, summary

# ---------- plots ----------


def plot_band_summary(summary_df):
    x = np.arange(len(summary_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, summary_df["RMSE"],
           width, label="RMSE", color="steelblue")
    ax.bar(x + width/2, summary_df["MAE"],  width, label="MAE",  color="navy")

    ax2 = ax.twinx()
    ax2.plot(
        x,
        summary_df["Bias (pred-true)"],
        marker="o",
        linestyle="--",
        label="Bias (pred-true)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["band_label"], rotation=10)
    ax.set_title("Performance by Price Band", fontweight="bold")
    ax.set_xlabel("Price Band", fontweight="bold")
    ax.set_ylabel("Error (€)", fontweight="bold")
    ax2.set_ylabel("Bias (€)", fontweight="bold")

    ax.yaxis.set_major_formatter(FuncFormatter(euro_fmt))
    ax2.yaxis.set_major_formatter(FuncFormatter(euro_fmt))

    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_abs_error_boxplot(df_banded):
    order = (
        df_banded[["band", "band_label"]]
        .dropna()
        .drop_duplicates()
        .sort_values("band")["band_label"]
        .tolist()
    )

    data = [df_banded.loc[df_banded["band_label"]
                          == b, "abs_err"].values for b in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title(
        "Absolute Error Distribution by Price Band (Holdout)", fontweight="bold")
    ax.set_xlabel("Price Band", fontweight="bold")
    ax.set_ylabel("Absolute Error (€)", fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(euro_fmt))
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()
