from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass
class DataPrepReport:
    rows_in: int
    rows_out: int
    dropped_missing: int


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Robust CSV loader:
    - tries default delimiter (comma)
    - if it loads as a single column, retries with ';'
    """
    path = Path(path)

    df = pd.read_csv(path)

    # If the CSV is actually semicolon-delimited, pandas may load 1 big column.
    # Note: if a true single-column CSV is expected, this heuristic may mis-detect.
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")

    # Clean column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df


def basic_checks(df: pd.DataFrame, target: str) -> dict[str, int | float | None]:
    """Small sanity checks to store in reports."""
    if target in df.columns:
        tmin = df[target].min()
        tmax = df[target].max()
        missing_target = int(df[target].isna().sum())
    else:
        tmin = None
        tmax = None
        missing_target = None

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_target": missing_target,
        "target_min": float(tmin) if tmin is not None else None,
        "target_max": float(tmax) if tmax is not None else None,
    }


def prepare_xy(
    df: pd.DataFrame, features: Sequence[str], target: str
) -> tuple[pd.DataFrame, pd.Series, DataPrepReport]:
    """
    Select X/y and drop rows with missing values in required columns.
    Minimum prep to avoid model crashes when estimators do not support NaNs.
    """
    required = list(features) + [target]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    rows_in = int(df.shape[0])

    sub = df[required].copy()
    sub_before = int(sub.shape[0])

    # Explicit subset makes intent clearer
    sub = sub.dropna(subset=required)
    rows_out = int(sub.shape[0])
    dropped = sub_before - rows_out

    X = sub[list(features)]
    y = sub[target]

    report = DataPrepReport(
        rows_in=rows_in, rows_out=rows_out, dropped_missing=dropped)
    return X, y, report
