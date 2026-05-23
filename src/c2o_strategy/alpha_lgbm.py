"""LightGBM walk-forward alpha model for comparison with linear baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig
from c2o_strategy.features import FEATURE_PRIORS


def _lgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def score_panel_lgbm(
    panel: pd.DataFrame,
    rank_columns: list[str],
    config: StrategyConfig,
    training_window: str = "expanding",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward LightGBM scoring using the same rank features.

    For each year Y, trains on all eligible data up to Dec 31 of Y-1
    (expanding window) or the most recent ``config.training_years``
    (rolling window). Scores year Y out-of-sample.

    Returns the scored panel and a diagnostics DataFrame.
    """
    if not _lgbm_available():
        raise ImportError("lightgbm is required for the tree-based alpha model")

    import lightgbm as lgb

    panel = panel.copy()
    panel["alpha_score_lgbm"] = np.nan

    # Vol-scaled target, same as the promoted linear strategy
    mask = panel["is_trade_eligible"] & panel["overnight_next"].notna()
    daily_vol = (panel["vol20"] / np.sqrt(252.0)).replace(0.0, np.nan)
    panel.loc[mask, "_target"] = (
        panel.loc[mask, "overnight_next"] / daily_vol[mask]
    ).replace([np.inf, -np.inf], np.nan)

    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 31,
        "max_depth": 6,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "min_child_samples": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": config.random_seed,
    }

    diag_rows: list[dict] = []
    frozen_model = None
    years = sorted(panel["year"].dropna().unique())

    for year in years:
        year = int(year)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        previous_year_end = year_start - pd.Timedelta(days=1)
        train_end = min(previous_year_end, config.development_cutoff_ts)

        if training_window == "expanding":
            train_start = config.start_ts
        else:
            train_start = train_end - pd.DateOffset(years=config.training_years)

        # After development cutoff, freeze the model
        if year_start > config.development_cutoff_ts and frozen_model is not None:
            model = frozen_model
            source = "frozen"
        else:
            train = panel.loc[
                panel["date"].between(train_start, train_end)
                & panel["is_trade_eligible"]
                & panel["_target"].notna()
            ]
            X_train = train[rank_columns].fillna(0.0).values
            y_train = train["_target"].values

            if len(X_train) < 20_000:
                # Not enough data — skip this year
                diag_rows.append({
                    "year": year, "n_train": len(X_train),
                    "source": "skipped", "feature_importance_top3": "",
                })
                continue

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            source = "trained"

            if year_start <= config.development_cutoff_ts:
                frozen_model = model

        year_mask = panel["year"].eq(year)
        X_score = panel.loc[year_mask, rank_columns].fillna(0.0).values
        panel.loc[year_mask, "alpha_score_lgbm"] = model.predict(X_score)

        # Feature importance diagnostics
        importances = pd.Series(
            model.feature_importances_, index=rank_columns
        ).sort_values(ascending=False)
        top3 = ", ".join(f"{k}({v})" for k, v in importances.head(3).items())

        diag_rows.append({
            "year": year,
            "n_train": int(len(X_train)) if source == "trained" else 0,
            "source": source,
            "feature_importance_top3": top3,
        })
        print(f"  LightGBM year {year}: {source}, top features: {top3}", flush=True)

    panel = panel.drop(columns=["_target"], errors="ignore")
    diagnostics = pd.DataFrame(diag_rows)
    return panel, diagnostics


def compute_lgbm_ic(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily Spearman IC for the LightGBM alpha score."""
    mask = (
        panel["is_trade_eligible"]
        & panel["alpha_score_lgbm"].notna()
        & panel["overnight_next"].notna()
    )
    rows: list[dict] = []
    for date, chunk in panel.loc[mask].groupby("date"):
        if len(chunk) < 25:
            continue
        ic = chunk["alpha_score_lgbm"].corr(chunk["overnight_next"], method="spearman")
        rows.append({"date": date, "ic_lgbm": float(ic), "n": int(len(chunk))})
    return pd.DataFrame(rows)


def blend_scores(
    panel: pd.DataFrame, linear_weight: float = 0.5
) -> pd.DataFrame:
    """Blend linear and LightGBM alpha scores."""
    panel = panel.copy()
    lgbm_weight = 1.0 - linear_weight
    has_both = panel["alpha_score"].notna() & panel["alpha_score_lgbm"].notna()
    panel.loc[has_both, "alpha_score_blend"] = (
        linear_weight * panel.loc[has_both, "alpha_score"]
        + lgbm_weight * panel.loc[has_both, "alpha_score_lgbm"]
    )
    # Where only one exists, use that one
    only_linear = panel["alpha_score"].notna() & panel["alpha_score_lgbm"].isna()
    panel.loc[only_linear, "alpha_score_blend"] = panel.loc[only_linear, "alpha_score"]
    only_lgbm = panel["alpha_score"].isna() & panel["alpha_score_lgbm"].notna()
    panel.loc[only_lgbm, "alpha_score_blend"] = panel.loc[only_lgbm, "alpha_score_lgbm"]
    return panel
