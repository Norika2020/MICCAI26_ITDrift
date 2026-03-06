"""Utilities for post-hoc risk scoring under synthetic and missing-modality shift.

This module pulls together the signal-fusion and evaluation code used in the
BraTS 2020 robustness monitoring experiments. The aim here is not to mirror the
notebook cell by cell, but to expose the same workflow through a small set of
functions that are easier to read and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES: List[str] = [
    "JS_input_mean",
    "InputEntropy_delta_mean",
    "PredEntropy_tumor_delta",
]

SYNTHETIC_DRIFT_FILES: Mapping[str, str] = {
    "brightness": "results_brightness_with_entropy_brats2020_test.csv",
    "noise": "results_noise_with_entropy_brats2020_test.csv",
    "bias": "results_bias_with_entropy_brats2020_test.csv",
}

DRIFT_PARAM_COLUMNS: Mapping[str, str] = {
    "brightness": "delta",
    "noise": "alpha",
    "bias": "beta",
}


@dataclass(frozen=True)
class RidgeSearchConfig:
    """Settings for ridge fusion."""

    alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv: int = 5
    scoring: str = "neg_mean_absolute_error"


@dataclass(frozen=True)
class CalibrationConfig:
    """Few-shot calibration setup for missing-modality runs."""

    calibration_size: int = 10
    n_repeats: int = 200
    ridge_alpha: float = 1.0
    random_seed: int = 1234


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _safe_corr(metric_fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 3:
        return np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(metric_fn(y_true, y_pred)[0])



def _replace_zero_iqr(iqr: pd.Series, eps: float = 1e-8) -> pd.Series:
    iqr = iqr.copy()
    return iqr.replace(0, eps)



def clean_numeric_frame(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    """Drop rows that cannot be used for modelling."""

    return (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(required_columns))
        .copy()
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_synthetic_shift_dataset(base_dir: str | Path) -> pd.DataFrame:
    """Build the synthetic-shift regression table.

    Expected files in ``base_dir``:
    - baseline_metrics_brats2020_test.csv
    - results_brightness_with_entropy_brats2020_test.csv
    - results_noise_with_entropy_brats2020_test.csv
    - results_bias_with_entropy_brats2020_test.csv
    """

    base_dir = Path(base_dir)
    clean_path = base_dir / "baseline_metrics_brats2020_test.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing baseline file: {clean_path}")

    clean_df = pd.read_csv(clean_path)
    drift_frames: List[pd.DataFrame] = []

    for drift_type, filename in SYNTHETIC_DRIFT_FILES.items():
        csv_path = base_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing synthetic-shift file: {csv_path}")

        drift_df = pd.read_csv(csv_path).copy()
        drift_df["drift_type"] = drift_type
        drift_param_col = DRIFT_PARAM_COLUMNS[drift_type]
        drift_df["drift_strength"] = drift_df[drift_param_col].astype(float)
        drift_df["drift_param"] = drift_param_col
        drift_frames.append(drift_df)

    drift_df = pd.concat(drift_frames, ignore_index=True)

    required_clean = {"case_id", "Dice_WT", "Dice_TC", "Dice_ET", "HD95_WT"}
    required_drift = {"case_id", "Dice_WT", "Dice_TC", "Dice_ET", "HD95_WT", "drift_type"}

    missing_clean = required_clean - set(clean_df.columns)
    missing_drift = required_drift - set(drift_df.columns)
    if missing_clean:
        raise ValueError(f"Baseline CSV is missing columns: {sorted(missing_clean)}")
    if missing_drift:
        raise ValueError(f"Synthetic CSVs are missing columns: {sorted(missing_drift)}")

    clean_df = clean_df.rename(
        columns={
            "Dice_WT": "Dice_WT_clean",
            "Dice_TC": "Dice_TC_clean",
            "Dice_ET": "Dice_ET_clean",
            "HD95_WT": "HD95_WT_clean",
        }
    ).drop_duplicates(subset=["case_id"])

    merged = drift_df.merge(clean_df, on="case_id", how="inner")
    merged["dDice_WT"] = merged["Dice_WT"] - merged["Dice_WT_clean"]
    merged["dHD95_WT"] = merged["HD95_WT"] - merged["HD95_WT_clean"]
    merged["sevDice_WT"] = -merged["dDice_WT"]
    merged["sevHD95_WT"] = merged["dHD95_WT"]
    merged["severity_level"] = (
        merged.groupby("drift_type")["drift_strength"].rank(method="dense").astype(int)
    )
    return merged



def load_missing_modality_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the missing-modality table and attach severity targets."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing missing-modality file: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    required = {
        "case_id",
        "missing_mod",
        "Dice_WT",
        "Dice_TC",
        "Dice_ET",
        "Dice_WT_clean",
        "Dice_TC_clean",
        "Dice_ET_clean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing-modality CSV is missing columns: {sorted(missing)}")

    if "dDice_WT" not in df.columns:
        df["dDice_WT"] = df["Dice_WT"] - df["Dice_WT_clean"]
    df["dDice_TC"] = df["Dice_TC"] - df["Dice_TC_clean"]
    df["dDice_ET"] = df["Dice_ET"] - df["Dice_ET_clean"]

    df["sevDice_WT"] = -df["dDice_WT"]
    df["sevDice_TC"] = -df["dDice_TC"]
    df["sevDice_ET"] = -df["dDice_ET"]
    return df


# -----------------------------------------------------------------------------
# Scoring and fusion
# -----------------------------------------------------------------------------

def fit_reference_stats(df: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
    """Reference median and IQR used by MeanZ."""

    med = df[list(features)].median()
    iqr = df[list(features)].quantile(0.75) - df[list(features)].quantile(0.25)
    return med, _replace_zero_iqr(iqr)



def score_single_feature(train_df: pd.DataFrame, test_df: pd.DataFrame, feature: str) -> np.ndarray:
    """Standardise one signal using the training split only."""

    scaler = StandardScaler()
    scaler.fit(train_df[[feature]].values)
    return scaler.transform(test_df[[feature]].values).ravel()



def score_meanz_from_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    """Parameter-free fusion using training-set robust statistics."""

    med, iqr = fit_reference_stats(train_df, features)
    z = (test_df[list(features)] - med) / iqr
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.mean(axis=1).to_numpy()



def score_meanz_from_reference(
    test_df: pd.DataFrame,
    features: Sequence[str],
    median_ref: pd.Series,
    iqr_ref: pd.Series,
) -> np.ndarray:
    """MeanZ against a fixed reference pool, typically the synthetic shift set."""

    z = (test_df[list(features)] - median_ref) / (iqr_ref + 1e-8)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.mean(axis=1).to_numpy()



def fit_predict_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    search: RidgeSearchConfig = RidgeSearchConfig(),
) -> Tuple[Pipeline, np.ndarray, Dict[str, float]]:
    """Fit ridge fusion with simple alpha tuning."""

    X_train = train_df[list(features)].to_numpy()
    y_train = train_df[target].to_numpy()
    X_test = test_df[list(features)].to_numpy()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={"ridge__alpha": list(search.alphas)},
        cv=search.cv,
        scoring=search.scoring,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.predict(X_test), dict(grid.best_params_)



def fit_fixed_ridge(
    train_df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    alpha: float,
) -> Pipeline:
    """Fit ridge fusion with a fixed alpha, used in few-shot calibration runs."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(train_df[list(features)], train_df[target])
    return model


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Main regression and ranking metrics used in the notebook."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson_r": _safe_corr(pearsonr, y_true, y_pred),
        "Spearman_rho": _safe_corr(spearmanr, y_true, y_pred),
        "Kendall_tau": _safe_corr(kendalltau, y_true, y_pred),
    }



def z_score_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE after standardising both vectors, useful when scales differ."""

    scaler_true = StandardScaler()
    scaler_pred = StandardScaler()
    y_true_z = scaler_true.fit_transform(np.asarray(y_true).reshape(-1, 1)).ravel()
    y_pred_z = scaler_pred.fit_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
    return float(mean_squared_error(y_true_z, y_pred_z))



def leave_one_drift_out_evaluation(
    synthetic_df: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    targets: Sequence[str] = ("sevDice_WT", "sevHD95_WT"),
    search: RidgeSearchConfig = RidgeSearchConfig(),
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame]]:
    """Evaluate severity prediction under synthetic shift using leave-one-drift-out splits.

    Returns
    -------
    results_df:
        Per-drift, per-target summary table.
    predictions:
        Enriched test-set frames keyed by ``(drift_type, target)``.
    """

    required = list(features) + list(targets) + ["drift_type"]
    df = clean_numeric_frame(synthetic_df, required)

    rows: List[Dict[str, float]] = []
    predictions: Dict[Tuple[str, str], pd.DataFrame] = {}
    drift_types = list(df["drift_type"].dropna().unique())

    for target in targets:
        for held_out_drift in drift_types:
            train_df = df[df["drift_type"] != held_out_drift].copy()
            test_df = df[df["drift_type"] == held_out_drift].copy()
            y_true = test_df[target].to_numpy()

            ridge_model, ridge_pred, best_params = fit_predict_ridge(
                train_df, test_df, features, target, search=search
            )
            test_df[f"pred_RidgeFusion_{target}"] = ridge_pred
            rows.append(
                {
                    "test_drift": held_out_drift,
                    "target": target,
                    "method": "RidgeFusion",
                    "n_test": len(test_df),
                    **best_params,
                    **regression_metrics(y_true, ridge_pred),
                }
            )

            meanz_pred = score_meanz_from_train(train_df, test_df, features)
            test_df[f"pred_MeanZ_{target}"] = meanz_pred
            rows.append(
                {
                    "test_drift": held_out_drift,
                    "target": target,
                    "method": "MeanZ",
                    "n_test": len(test_df),
                    **regression_metrics(y_true, meanz_pred),
                }
            )

            for feature in features:
                feature_pred = score_single_feature(train_df, test_df, feature)
                test_df[f"pred_Single_{feature}_{target}"] = feature_pred
                rows.append(
                    {
                        "test_drift": held_out_drift,
                        "target": target,
                        "method": f"Single:{feature}",
                        "n_test": len(test_df),
                        **regression_metrics(y_true, feature_pred),
                    }
                )

            predictions[(held_out_drift, target)] = test_df

    results_df = pd.DataFrame(rows).sort_values(["target", "test_drift", "method"]).reset_index(drop=True)
    return results_df, predictions



def summarise_missing_modality_signals(
    missing_df: pd.DataFrame,
    synthetic_reference_df: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    calibration: CalibrationConfig = CalibrationConfig(),
    targets: Sequence[Tuple[str, str]] = (("WT", "sevDice_WT"), ("TC", "sevDice_TC"), ("ET", "sevDice_ET")),
) -> pd.DataFrame:
    """Summarise missing-modality performance for single signals, MeanZ and few-shot ridge.

    Ridge is calibrated within each missing-modality subset using ``K`` labelled cases,
    then evaluated on the remaining cases. Results are averaged across repeated random splits.
    """

    miss = clean_numeric_frame(missing_df, list(features) + [t for _, t in targets] + ["missing_mod"])
    med_ref, iqr_ref = fit_reference_stats(synthetic_reference_df, features)
    rng = np.random.default_rng(calibration.random_seed)

    rows: List[Dict[str, float]] = []
    method_labels = ["JS", "InputEntropy", "PredEntropy", "MeanZ", f"Ridge(K={calibration.calibration_size})"]

    for missing_mod in sorted(miss["missing_mod"].dropna().unique()):
        df_mod = miss[miss["missing_mod"] == missing_mod].reset_index(drop=True)
        n_cases = len(df_mod)
        if calibration.calibration_size >= n_cases - 2:
            raise ValueError(
                f"Calibration size K={calibration.calibration_size} is too large for {missing_mod} (n={n_cases})."
            )

        for target_name, target_col in targets:
            per_method: MutableMapping[str, List[float]] = {label: [] for label in method_labels}
            per_method_zmse: MutableMapping[str, List[float]] = {label: [] for label in method_labels}

            for _ in range(calibration.n_repeats):
                perm = rng.permutation(n_cases)
                calib_idx = perm[: calibration.calibration_size]
                test_idx = perm[calibration.calibration_size :]
                calib_df = df_mod.iloc[calib_idx]
                test_df = df_mod.iloc[test_idx]
                y_true = test_df[target_col].to_numpy()

                predictions = {
                    "JS": test_df["JS_input_mean"].to_numpy(),
                    "InputEntropy": test_df["InputEntropy_delta_mean"].to_numpy(),
                    "PredEntropy": test_df["PredEntropy_tumor_delta"].to_numpy(),
                    "MeanZ": score_meanz_from_reference(test_df, features, med_ref, iqr_ref),
                }

                ridge_model = fit_fixed_ridge(
                    calib_df,
                    target=target_col,
                    features=features,
                    alpha=calibration.ridge_alpha,
                )
                predictions[f"Ridge(K={calibration.calibration_size})"] = ridge_model.predict(test_df[list(features)])

                for method_name, y_pred in predictions.items():
                    per_method[method_name].append(_safe_corr(spearmanr, y_true, y_pred))
                    per_method_zmse[method_name].append(z_score_mse(y_true, y_pred))

            for method_name in method_labels:
                rows.append(
                    {
                        "missing_mod": missing_mod,
                        "target": target_name,
                        "method": method_name,
                        "K": calibration.calibration_size,
                        "n_repeats": calibration.n_repeats,
                        "Spearman_mean": float(np.nanmean(per_method[method_name])),
                        "Spearman_std": float(np.nanstd(per_method[method_name])),
                        "zMSE_mean": float(np.nanmean(per_method_zmse[method_name])),
                        "zMSE_std": float(np.nanstd(per_method_zmse[method_name])),
                    }
                )

    return pd.DataFrame(rows).sort_values(["missing_mod", "target", "method"]).reset_index(drop=True)



def calibration_curve_missing_modality(
    missing_df: pd.DataFrame,
    synthetic_reference_df: pd.DataFrame,
    k_values: Iterable[int],
    features: Sequence[str] = DEFAULT_FEATURES,
    n_repeats: int = 200,
    ridge_alpha: float = 1.0,
    random_seed: int = 1234,
    targets: Sequence[Tuple[str, str]] = (("WT", "sevDice_WT"), ("TC", "sevDice_TC"), ("ET", "sevDice_ET")),
) -> pd.DataFrame:
    """Sweep calibration size ``K`` for missing-modality ridge fusion."""

    miss = clean_numeric_frame(missing_df, list(features) + [t for _, t in targets] + ["missing_mod"])
    med_ref, iqr_ref = fit_reference_stats(synthetic_reference_df, features)
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, float]] = []

    for missing_mod in sorted(miss["missing_mod"].dropna().unique()):
        df_mod = miss[miss["missing_mod"] == missing_mod].reset_index(drop=True)
        n_cases = len(df_mod)

        for k in k_values:
            if k >= n_cases - 2:
                continue

            for target_name, target_col in targets:
                scores = {
                    "JS": [],
                    "InputEntropy": [],
                    "PredEntropy": [],
                    "MeanZ": [],
                    "Ridge": [],
                }

                for _ in range(n_repeats):
                    perm = rng.permutation(n_cases)
                    calib_df = df_mod.iloc[perm[:k]]
                    test_df = df_mod.iloc[perm[k:]]
                    y_true = test_df[target_col].to_numpy()

                    scores["JS"].append(_safe_corr(spearmanr, y_true, test_df["JS_input_mean"].to_numpy()))
                    scores["InputEntropy"].append(
                        _safe_corr(spearmanr, y_true, test_df["InputEntropy_delta_mean"].to_numpy())
                    )
                    scores["PredEntropy"].append(
                        _safe_corr(spearmanr, y_true, test_df["PredEntropy_tumor_delta"].to_numpy())
                    )
                    scores["MeanZ"].append(
                        _safe_corr(
                            spearmanr,
                            y_true,
                            score_meanz_from_reference(test_df, features, med_ref, iqr_ref),
                        )
                    )

                    ridge_model = fit_fixed_ridge(
                        calib_df,
                        target=target_col,
                        features=features,
                        alpha=ridge_alpha,
                    )
                    scores["Ridge"].append(
                        _safe_corr(spearmanr, y_true, ridge_model.predict(test_df[list(features)]))
                    )

                for method_name, values in scores.items():
                    rows.append(
                        {
                            "missing_mod": missing_mod,
                            "target": target_name,
                            "method": method_name,
                            "K": int(k),
                            "n_repeats": int(n_repeats),
                            "Spearman_mean": float(np.nanmean(values)),
                            "Spearman_std": float(np.nanstd(values)),
                        }
                    )

    return pd.DataFrame(rows).sort_values(["missing_mod", "target", "K", "method"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------------

def run_full_risk_scoring_pipeline(
    base_dir: str | Path,
    missing_modality_csv: str | Path,
    features: Sequence[str] = DEFAULT_FEATURES,
    calibration: CalibrationConfig = CalibrationConfig(),
    ridge_search: RidgeSearchConfig = RidgeSearchConfig(),
    calibration_k_values: Optional[Iterable[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the core tables used in the notebook.

    This keeps the main entry point simple for a reproduction script:

    - build the synthetic-shift table
    - evaluate leave-one-drift-out severity prediction
    - summarise missing-modality ranking performance
    - optionally sweep calibration size ``K``
    """

    synthetic_df = load_synthetic_shift_dataset(base_dir)
    synthetic_results, _ = leave_one_drift_out_evaluation(
        synthetic_df,
        features=features,
        search=ridge_search,
    )

    missing_df = load_missing_modality_dataset(missing_modality_csv)
    missing_results = summarise_missing_modality_signals(
        missing_df,
        synthetic_reference_df=synthetic_df,
        features=features,
        calibration=calibration,
    )

    outputs: Dict[str, pd.DataFrame] = {
        "synthetic_shift_results": synthetic_results,
        "missing_modality_results": missing_results,
    }

    if calibration_k_values is not None:
        outputs["missing_modality_calibration_curve"] = calibration_curve_missing_modality(
            missing_df,
            synthetic_reference_df=synthetic_df,
            k_values=calibration_k_values,
            features=features,
            n_repeats=calibration.n_repeats,
            ridge_alpha=calibration.ridge_alpha,
            random_seed=calibration.random_seed,
        )

    return outputs
