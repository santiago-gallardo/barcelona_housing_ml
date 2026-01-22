import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(
    df: pd.DataFrame,
    target: str,
    save_figs: bool = False,
    nb_save=None,
    filename: str = "target_distribution_notebook.png",
    bins: int = 60,
):
    """
    Histogram of target distribution.
    If save_figs=True, expects nb_save(fig, filename, dpi=...) callable.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(
        df[target].dropna().values,
        bins=bins,
        alpha=0.75,
        color="steelblue",
        edgecolor="navy",
        linewidth=0.6,
    )

    ax.set_title("Target Distribution: PRICE",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Price (€)", fontsize=12, fontweight="bold", labelpad=15)
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold", labelpad=15)

    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)

    fig.tight_layout()

    if save_figs and nb_save is not None:
        nb_save(fig, filename, dpi=300)

    plt.show()
    return fig


def plot_topk_abs_corr_with_target(
    df: pd.DataFrame,
    target: str,
    topk: int = 10,
    save_figs: bool = False,
    nb_save=None,
    filename: str = "top10_corr_notebook.png",
):
    """
    Horizontal bar plot of top-k absolute correlations with target.
    Uses df.corr(numeric_only=True).
    """
    corr = (
        df.corr(numeric_only=True)[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )
    top = corr.head(topk)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(
        top.index[::-1],
        top.values[::-1],
        color="steelblue",
        edgecolor="navy",
        alpha=0.8,
        height=0.6,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="navy",
        )

    ax.set_xlabel("Absolute Correlation", fontsize=12,
                  fontweight="bold", labelpad=15)
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold", labelpad=15)
    ax.set_title(
        f"Top-{topk} Features by Absolute Correlation with PRICE",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_figs and nb_save is not None:
        nb_save(fig, filename, dpi=300)

    plt.show()
    return fig


def plot_price_vs_constructed_area(
    df: pd.DataFrame,
    target: str,
    area_col: str = "CONSTRUCTEDAREA",
    xlim=(15, 1000),
    save_figs: bool = False,
    nb_save=None,
    filename: str = "price_vs_area_notebook.png",
):
    """
    Scatter plot: target vs constructed area.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(
        df[area_col],
        df[target],
        s=25,
        alpha=0.35,
        color="steelblue",
        edgecolors="navy",
        linewidth=0.3,
    )

    ax.set_title("Price vs Constructed Area",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Constructed Area (m²)", fontsize=12,
                  fontweight="bold", labelpad=15)
    ax.set_ylabel("Price (€)", fontsize=12, fontweight="bold", labelpad=15)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)

    fig.tight_layout()

    if save_figs and nb_save is not None:
        nb_save(fig, filename, dpi=300)

    plt.show()
    return fig


def run_basic_eda_plots(
    df: pd.DataFrame,
    target: str,
    save_figs: bool = False,
    nb_save=None,
):
    """
    Convenience wrapper: runs the 3 plots with notebook-style saving.
    """
    plot_target_distribution(df, target, save_figs=save_figs, nb_save=nb_save)
    plot_topk_abs_corr_with_target(
        df, target, topk=10, save_figs=save_figs, nb_save=nb_save)
    plot_price_vs_constructed_area(
        df, target, save_figs=save_figs, nb_save=nb_save)
