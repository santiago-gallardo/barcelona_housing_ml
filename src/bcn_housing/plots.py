from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_savefig(
    fig: plt.Figure,
    out_path: str | Path,
    *,
    dpi: int = 300,
    tight: bool = True,
    pad_inches: float = 0.1,
) -> None:
    """
    Save a figure defensively.

    Tries:
    1) dpi=300 with tight bbox (best look)
    2) dpi=300 without tight bbox (often avoids huge render size)
    3) lower dpi without tight bbox as last resort
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Best attempt
    try:
        if tight:
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                        pad_inches=pad_inches)
        else:
            fig.savefig(out_path, dpi=dpi)
        return
    except MemoryError:
        pass

    # 2) Same DPI, disable tight bbox (often fixes MemoryError)
    try:
        fig.savefig(out_path, dpi=dpi)
        return
    except MemoryError:
        pass

    # 3) Last resort: reduce DPI (still save something)
    fig.savefig(out_path, dpi=150)


def save_feature_importance(
    model,
    feature_names,
    out_path: str | Path | None = None,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    top_k: int | None = 30,
    dpi: int = 200,
    tight: bool = True,
    save: bool = True,
    show: bool = False,
    close: bool = True,
    return_fig: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Horizontal bar chart of feature importances.

    Notes
    -----
    - `top_k`: if not None, keep only the top-k most important features to avoid
      huge plots (and MemoryError on save).
    - Uses `_safe_savefig` to avoid Matplotlib MemoryError on large/tight renders.
    """
    reg = getattr(model, "regressor_", model)
    if not hasattr(reg, "feature_importances_"):
        raise ValueError(
            "Model does not expose feature_importances_ attribute")

    importances = np.asarray(reg.feature_importances_, dtype=float)

    if len(feature_names) != len(importances):
        raise ValueError(
            f"feature_names ({len(feature_names)}) != importances ({len(importances)}). "
            "Make sure you pass the SAME feature list used to train the model."
        )

    importance_df = (
        pd.DataFrame({"Feature": list(feature_names),
                     "Importance": importances})
        .sort_values("Importance", ascending=True)
    )

    # Keep only top-k (largest) importances if requested
    if top_k is not None and len(importance_df) > top_k:
        importance_df = importance_df.tail(top_k)

    n = len(importance_df)
    fig_h = max(4.5, 0.35 * n + 1.5)  # dynamic height so labels don't explode
    fig, ax = plt.subplots(figsize=(14, fig_h))

    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        alpha=0.85,
        height=0.6,
    )

    # Annotate values only when the plot isn't too crowded
    if n <= 30:
        max_imp = float(np.nanmax(importances)) if len(importances) else 1.0
        offset = max_imp * \
            0.015 if np.isfinite(max_imp) and max_imp > 0 else 0.01
        for bar in bars:
            width = float(bar.get_width())
            ax.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if title is None:
        title = "Feature Importance"

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.90, subtitle, ha="center",
                 va="top", fontsize=12, color="black")

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save and out_path is not None:
        _safe_savefig(fig, out_path, dpi=dpi, tight=tight, pad_inches=0.1)

    if show:
        plt.show()

    if close and not return_fig:
        plt.close(fig)

    return (importance_df, fig, ax) if return_fig else importance_df


def save_ytrue_vs_ypred(
    y_true,
    y_pred,
    out_path: str | Path | None = None,
    *,
    add_identity_line: bool = True,
    identity_line_label: str = "Ideal line (y = x)",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    equal_axes: bool = True,
    dpi: int = 200,
    tight: bool = True,
    save: bool = True,
    show: bool = False,
    close: bool = True,
    return_fig: bool = False,
) -> None | tuple[plt.Figure, plt.Axes]:
    """Scatter plot of predicted vs true values with optional identity line."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.45,
        s=30,
        edgecolors="navy",
        linewidth=0.3,
    )

    if add_identity_line:
        if xlim is not None:
            mn, mx = xlim
        else:
            mn = float(np.nanmin(y_true))
            mx = float(np.nanmax(y_true))

        if ylim is not None:
            mn = min(mn, float(ylim[0]))
            mx = max(mx, float(ylim[1]))

        ax.plot(
            [mn, mx],
            [mn, mx],
            linestyle="--",
            linewidth=2.5,
            color="crimson",
            label=identity_line_label,
            alpha=0.9,
        )
        ax.legend(fontsize=12, loc="upper left", framealpha=0.95)

    ax.set_title("Model Validation: True vs Predicted Values",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("True Price (€)", fontsize=13,
                  fontweight="bold", labelpad=15)
    ax.set_ylabel("Predicted Price (€)", fontsize=13,
                  fontweight="bold", labelpad=15)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if equal_axes:
        ax.set_aspect("equal", adjustable="box")

    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save and out_path is not None:
        _safe_savefig(fig, out_path, dpi=dpi, tight=tight, pad_inches=0.1)

    if show:
        plt.show()

    if close and not return_fig:
        plt.close(fig)

    if return_fig:
        return fig, ax
    return None


def save_residuals_hist_log(
    y_true,
    y_pred,
    out_path: str | Path | None = None,
    *,
    bins: int = 50,
    xlim: tuple[float, float] = (-1.2, 1.2),
    dpi: int = 200,
    tight: bool = True,
    save: bool = True,
    show: bool = False,
    close: bool = True,
    return_fig: bool = False,
) -> None | tuple[plt.Figure, plt.Axes]:
    """Histogram of residuals in log space: log1p(y_true) - log1p(y_pred)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Defensive: avoid log1p domain errors
    y_true = np.clip(y_true, a_min=0.0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    log_res = np.log1p(y_true) - np.log1p(y_pred)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.hist(
        log_res,
        bins=bins,
        alpha=0.75,
        edgecolor="navy",
        linewidth=0.5,
    )

    ax.axvline(
        0,
        color="crimson",
        linestyle="--",
        linewidth=2.5,
        label="Residual = 0",
        alpha=0.9,
    )

    ax.set_title("Distribution of Residuals (Log Scale)",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Residual", fontsize=13, fontweight="bold", labelpad=15)
    ax.set_ylabel("Frequency", fontsize=13, fontweight="bold", labelpad=15)
    ax.set_xlim(*xlim)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95)

    fig.tight_layout()

    if save and out_path is not None:
        _safe_savefig(fig, out_path, dpi=dpi, tight=tight, pad_inches=0.1)

    if show:
        plt.show()

    if close and not return_fig:
        plt.close(fig)

    if return_fig:
        return fig, ax
    return None


def save_metric_evolution(
    results_df: pd.DataFrame,
    model_order: list[str],
    model_labels: dict[str, str],
    metric_col: str,
    out_path: str | Path | None = None,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    dpi: int = 200,
    tight: bool = True,
    save: bool = True,
    show: bool = False,
    close: bool = True,
    return_fig: bool = False,
) -> None | tuple[plt.Figure, plt.Axes]:
    """Line plot of a metric across models in a fixed order."""
    df = results_df.copy()
    df = df[df["Model"].isin(model_order)].copy()
    df["Model"] = pd.Categorical(
        df["Model"], categories=model_order, ordered=True)
    df = df.sort_values("Model")

    x_labels = [model_labels[m] for m in df["Model"].tolist()]
    y_vals = df[metric_col].astype(float).values

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_labels,
        y_vals,
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=8,
        alpha=0.8,
    )

    # Label offset based on data range (more stable than y*0.02)
    yr = float(np.nanmax(y_vals) - np.nanmin(y_vals)) if len(y_vals) else 1.0
    if not np.isfinite(yr) or yr == 0:
        yr = 1.0
    offset = yr * 0.02

    for x, y in zip(range(len(x_labels)), y_vals):
        ax.text(
            x,
            y + offset,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", labelpad=15)
    ax.set_xlabel("")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(11)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)

    fig.tight_layout()

    if save and out_path is not None:
        _safe_savefig(fig, out_path, dpi=dpi, tight=tight, pad_inches=0.1)

    if show:
        plt.show()

    if close and not return_fig:
        plt.close(fig)

    if return_fig:
        return fig, ax
    return None
