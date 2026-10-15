"""Reproducible dataset splitting for InvestWise NLP experiments.

Each ML task receives physically separate training, conformal
calibration, and locked test sets.

Business governance fixtures and deterministic control examples are
excluded from these splits.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_three_way_split(
    source_path: str | Path,
    output_directory: str | Path,
    prefix: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, calibration, and test partitions.

    Approximately 60 percent is used for model training.
    Approximately 20 percent is used only for conformal calibration.
    Approximately 20 percent remains locked for final evaluation.
    """

    source_path = Path(source_path)
    output_directory = Path(output_directory)

    df = pd.read_csv(source_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{source_path} must contain columns "
            f"{sorted(required_columns)}."
        )

    if df["text"].isna().any():
        raise ValueError(
            f"{source_path} contains missing text."
        )

    if df["label"].isna().any():
        raise ValueError(
            f"{source_path} contains missing labels."
        )

    if df["text"].duplicated().any():
        raise ValueError(
            f"{source_path} contains duplicate text."
        )

    train, remainder = train_test_split(
        df,
        test_size=0.40,
        random_state=random_state,
        stratify=df["label"],
    )

    calibration, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder["label"],
    )

    train = train.reset_index(drop=True)
    calibration = calibration.reset_index(drop=True)
    test = test.reset_index(drop=True)

    _assert_no_overlap(
        train,
        calibration,
        test,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        output_directory / f"{prefix}_train.csv",
        index=False,
    )

    calibration.to_csv(
        output_directory / f"{prefix}_calibration.csv",
        index=False,
    )

    test.to_csv(
        output_directory / f"{prefix}_test.csv",
        index=False,
    )

    return train, calibration, test


def _assert_no_overlap(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Fail immediately if text leaks between partitions."""

    train_text = set(train["text"])
    calibration_text = set(calibration["text"])
    test_text = set(test["text"])

    if train_text & calibration_text:
        raise RuntimeError(
            "training and calibration data overlap."
        )

    if train_text & test_text:
        raise RuntimeError(
            "training and test data overlap."
        )

    if calibration_text & test_text:
        raise RuntimeError(
            "calibration and test data overlap."
        )
