import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def short_label(model_name: str) -> str:
    name = str(model_name).lower()
    if "randomforest" in name and "10" in name:
        return "RF10"
    if "randomforest" in name and "5" in name:
        return "RF5+log"
    if "linear" in name or name.startswith("lr"):
        return "LR"
    if "decisiontree" in name:
        return "Tree"
    if "dummy" in name or "benchmark" in name:
        return "Dummy"
    return str(model_name)


def plot_cv_tradeoff(
    cv_sorted: pd.DataFrame,
    best_cv,
    final_cv=None,
    final_name: str | None = None,
    save_figs: bool = False,
    nb_save=None,
    filename: str = "tradeoff_cv_notebook.png",
    show: bool = True,
):
    """
    Trade-off plot: CV RMSE_mean vs #features with error bars.
    - cv_sorted: DataFrame with columns: Model, n_features, RMSE_mean, RMSE_std
    - best_cv/final_cv: row-like (pd.Series or dict) with keys Model, n_features, RMSE_mean
    - If save_figs=True, expects nb_save(fig, filename, dpi=...) callable.
    - If show=True -> shows plot and returns None (avoids duplicated render in notebooks).
      If show=False -> returns fig.
    """
    df_plot = cv_sorted.copy()

    x = df_plot["n_features"].astype(int).values
    y = df_plot["RMSE_mean"].astype(float).values
    yerr = df_plot["RMSE_std"].astype(float).values

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o",
        capsize=5, capthick=2,
        color="steelblue", ecolor="navy",
        markersize=8,
        linewidth=2, elinewidth=1.8,
        alpha=0.7
    )

    ax.set_title(
        "Model Trade-off: Complexity (# features) vs Performance (RMSE)",
        fontsize=15, fontweight="bold", pad=20
    )
    ax.set_xlabel("Number of Features", fontsize=12,
                  fontweight="bold", labelpad=12)
    ax.set_ylabel("RMSE (€) — CV Mean ± Std", fontsize=12,
                  fontweight="bold", labelpad=12)

    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8, color="gray")
    ax.set_axisbelow(True)

    for _, r in df_plot.iterrows():
        ax.annotate(
            short_label(r["Model"]),
            (int(r["n_features"]), float(r["RMSE_mean"])),
            textcoords="offset points",
            xytext=(0, 12),
            fontsize=10,
            fontweight="bold",
            ha="center",
            color="navy",
        )

    def _get(d, k):
        return d[k] if isinstance(d, dict) else d[k]

    best_model = str(_get(best_cv, "Model"))

    def highlight(model_name: str, color: str, marker: str = "o", label: str = ""):
        rr = df_plot[df_plot["Model"] == model_name]
        if not rr.empty:
            ax.scatter(
                int(rr["n_features"].iloc[0]),
                float(rr["RMSE_mean"].iloc[0]),
                s=200,
                facecolors="none",
                edgecolors=color,
                linewidths=2.5,
                marker=marker,
                label=label,
                zorder=5,
            )

    highlight(best_model, color="navy", marker="o", label="Best RMSE (RF-10)")

    if final_name is not None:
        highlight(str(final_name), color="crimson",
                  marker="s", label="Our choice (RF-5+log)")
    elif final_cv is not None:
        highlight(str(_get(final_cv, "Model")), color="crimson",
                  marker="s", label="Our choice")

    ax.legend(loc="upper right", fontsize=10, framealpha=0.95,
              edgecolor="navy", fancybox=True)

    fig.tight_layout()

    if save_figs and nb_save is not None:
        nb_save(fig, filename, dpi=300)

    if show:
        plt.show()
        return None

    return fig


def build_model_selection_markdown(best_cv, final_cv):
    """
    Returns a Markdown string with the trade-off summary.
    Expects best_cv and final_cv to have keys: RMSE_mean, n_features.
    """
    def _get(d, k):
        return d[k] if isinstance(d, dict) else d[k]

    if final_cv is None:
        return "### Model Selection Justification\n\nFinal model not available."

    rmse_diff = float(_get(final_cv, "RMSE_mean") - _get(best_cv, "RMSE_mean"))
    rmse_pct = (rmse_diff / float(_get(best_cv, "RMSE_mean"))) * 100
    features_diff = int(_get(final_cv, "n_features") -
                        _get(best_cv, "n_features"))

    md = f"""### Model Selection Justification

**Trade-off Analysis:**
- **RF(10)** achieves the best CV RMSE (€{float(_get(best_cv, 'RMSE_mean')):,.0f}), but requires **{int(_get(best_cv, 'n_features'))} features**
- **RF(5+log)** trades **+€{rmse_diff:,.0f} RMSE (+{rmse_pct:.1f}%)** for **{abs(features_diff)} fewer features**
- **Result:** Simpler, more maintainable model with acceptable performance loss

**Why RF(5+log)?** Fewer features reduce overfitting risk and make the model easier to explain in production.
"""
    return md
