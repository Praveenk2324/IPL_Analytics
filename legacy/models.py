"""
IPL Analytics — Machine Learning Models.

Implements a First Innings Score Predictor using a Random Forest
Regressor with scikit-learn pipelines.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib


# ===================================================================
# Feature Engineering
# ===================================================================

def prepare_ml_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Build a training dataset from merged ball-by-ball data.

    For each match, at every over boundary (overs >= 5), compute:
      - ``current_score``: cumulative runs at that over.
      - ``wickets_fallen``: cumulative wickets at that over.
      - ``overs_completed``: the over number.
      - ``venue``: match venue.
      - ``batting_team`` / ``bowling_team``.
      - ``final_score`` (target): total first-innings score.

    Only first-innings data is used.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged deliveries + match data.

    Returns
    -------
    pd.DataFrame
        Training-ready DataFrame with feature columns and target.
    """
    # Filter to first innings only
    first_innings = merged[merged["inning"] == 1].copy()

    # Compute final score per match
    final_scores = (
        first_innings.groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "final_score"})
    )

    # Cumulative score and wickets at each ball
    first_innings = first_innings.sort_values(["match_id", "over", "ball"])
    first_innings["cum_runs"] = first_innings.groupby("match_id")[
        "total_runs"
    ].cumsum()
    first_innings["cum_wickets"] = first_innings.groupby("match_id")[
        "is_wicket"
    ].cumsum()

    # Get state at end of each over (over >= 5 for meaningful predictions)
    over_end = (
        first_innings.groupby(["match_id", "over"])
        .agg(
            current_score=("cum_runs", "last"),
            wickets_fallen=("cum_wickets", "last"),
            venue=("venue", "first"),
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
        )
        .reset_index()
    )

    # Overs completed = over + 1 (0-indexed → 1-indexed)
    over_end["overs_completed"] = over_end["over"] + 1

    # Filter: overs 5–19 (after powerplay starts, before innings end)
    over_end = over_end[
        (over_end["overs_completed"] >= 5) & (over_end["overs_completed"] <= 20)
    ]

    # Merge final score
    over_end = over_end.merge(final_scores, on="match_id", how="left")

    # Select columns
    features = over_end[
        [
            "current_score",
            "wickets_fallen",
            "overs_completed",
            "venue",
            "batting_team",
            "bowling_team",
            "final_score",
        ]
    ].dropna()

    return features.reset_index(drop=True)


# ===================================================================
# Model Training
# ===================================================================

def train_model(
    features_df: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """Train a Random Forest Regressor for first-innings score prediction.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of ``prepare_ml_features``.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Seed for reproducibility.
    test_size : float
        Fraction of data held out for testing.

    Returns
    -------
    dict
        Keys: model, encoders, metrics, feature_names.
        - ``model``: trained RandomForestRegressor.
        - ``encoders``: dict of LabelEncoders for categorical columns.
        - ``metrics``: dict with MAE, RMSE, R2 on the test set.
        - ``feature_names``: list of feature column names.
    """
    df = features_df.copy()

    # Encode categorical features
    encoders = {}
    for col in ["venue", "batting_team", "bowling_team"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "current_score",
        "wickets_fallen",
        "overs_completed",
        "venue",
        "batting_team",
        "bowling_team",
    ]

    X = df[feature_cols]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "R2": round(r2_score(y_test, y_pred), 4),
    }

    return {
        "model": model,
        "encoders": encoders,
        "metrics": metrics,
        "feature_names": feature_cols,
    }


# ===================================================================
# Prediction
# ===================================================================

def predict_score(
    model: RandomForestRegressor,
    encoders: dict,
    current_score: int,
    wickets_fallen: int,
    overs_completed: float,
    venue: str,
    batting_team: str,
    bowling_team: str,
) -> int:
    """Predict the final first-innings score.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model.
    encoders : dict
        Label encoders for categorical columns.
    current_score : int
        Runs scored so far.
    wickets_fallen : int
        Wickets lost so far.
    overs_completed : float
        Overs bowled so far.
    venue : str
        Match venue.
    batting_team : str
        Batting team name.
    bowling_team : str
        Bowling team name.

    Returns
    -------
    int
        Predicted final first-innings score (rounded).
    """
    # Encode categoricals — handle unseen labels gracefully
    def safe_encode(encoder: LabelEncoder, value: str) -> int:
        """Encode a value, defaulting to 0 for unseen labels."""
        try:
            return int(encoder.transform([value])[0])
        except ValueError:
            return 0

    venue_enc = safe_encode(encoders["venue"], venue)
    bat_enc = safe_encode(encoders["batting_team"], batting_team)
    bowl_enc = safe_encode(encoders["bowling_team"], bowling_team)

    X = pd.DataFrame(
        [
            {
                "current_score": current_score,
                "wickets_fallen": wickets_fallen,
                "overs_completed": overs_completed,
                "venue": venue_enc,
                "batting_team": bat_enc,
                "bowling_team": bowl_enc,
            }
        ]
    )

    prediction = model.predict(X)[0]
    return max(int(round(prediction)), current_score)  # can't be less than current


# ===================================================================
# Persistence
# ===================================================================

def save_model(
    model_dict: dict,
    path: str = "models/score_predictor.joblib",
) -> None:
    """Save trained model and encoders to disk.

    Parameters
    ----------
    model_dict : dict
        Output of ``train_model``.
    path : str
        File path for the joblib dump.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_dict, path)


def load_model(path: str = "models/score_predictor.joblib") -> dict:
    """Load a previously saved model from disk.

    Parameters
    ----------
    path : str
        File path to the joblib file.

    Returns
    -------
    dict
        Same structure as ``train_model`` output.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Train and save the model first."
        )
    return joblib.load(path)
