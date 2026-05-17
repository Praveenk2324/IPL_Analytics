"""
IPL Analytics — Machine Learning Models.

Four models:
1. Logistic Regression — baseline win predictor (match-level features)
2. Random Forest — ball-by-ball win probability with non-linear patterns
3. XGBoost — primary win probability model with complex interactions
4. K-Means Clustering — player performance segmentation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. XGBoost model will be skipped.")


# ===================================================================
# Feature Preparation
# ===================================================================

def prepare_match_level_features(matches: pd.DataFrame) -> tuple:
    """Prepare match-level features for Logistic Regression.

    Features: venue, toss_decision, team1, team2, first_innings_score, toss_win_is_match_win
    Target: did batting_first_team win? (binary)

    Returns
    -------
    tuple
        (X_df, y_series, encoders_dict)
    """
    df = matches[matches["winner"] != "No Result"].copy()

    # Need batting_first_team and first_innings_score
    required = ["venue", "toss_decision", "batting_first_team",
                 "batting_second_team", "first_innings_score", "winner"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Target: did the team batting first win?
    df["batting_first_won"] = (df["winner"] == df["batting_first_team"]).astype(int)

    # Encode categoricals
    encoders = {}
    cat_cols = ["venue", "toss_decision", "batting_first_team", "batting_second_team"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c + "_enc" for c in cat_cols] + ["first_innings_score"]
    X = df[feature_cols]
    y = df["batting_first_won"]

    return X, y, encoders


def prepare_ball_level_features(ball_features: pd.DataFrame) -> tuple:
    """Prepare ball-level features for RF and XGBoost win probability models.

    Features: cum_runs, cum_wickets, overs_completed, current_run_rate,
              target, runs_remaining, balls_remaining, required_run_rate,
              match_phase (encoded)
    Target: batting_team_won (binary)

    Returns
    -------
    tuple
        (X_df, y_series)
    """
    df = ball_features.copy()

    # Sample every 6th ball (end of over) to reduce dataset size & redundancy
    df = df.groupby("match_id").apply(
        lambda g: g.iloc[5::6] if len(g) > 6 else g.iloc[-1:]
    ).reset_index(drop=True)

    # Encode match_phase
    phase_map = {"powerplay": 0, "middle": 1, "death": 2}
    df["match_phase_enc"] = df["match_phase"].map(phase_map).fillna(1).astype(int)

    feature_cols = [
        "cum_runs", "cum_wickets", "overs_completed",
        "current_run_rate", "target", "runs_remaining",
        "balls_remaining", "required_run_rate", "match_phase_enc",
    ]

    df = df.dropna(subset=feature_cols + ["batting_team_won"])

    # Cap extreme required run rates
    df["required_run_rate"] = df["required_run_rate"].clip(upper=36.0)

    X = df[feature_cols]
    y = df["batting_team_won"]

    return X, y


def prepare_player_clustering_features(
    batsman_stats_df: pd.DataFrame,
    bowler_stats_df: pd.DataFrame,
    min_bat_balls: int = 200,
    min_bowl_balls: int = 200,
) -> tuple:
    """Prepare player features for K-Means clustering.

    Returns
    -------
    tuple
        (bat_features, bowl_features, bat_scaler, bowl_scaler)
    """
    # === Batsmen ===
    bat = batsman_stats_df[batsman_stats_df["balls_faced"] >= min_bat_balls].copy()
    bat_feature_cols = ["strike_rate", "average", "boundary_pct"]
    bat_features = bat[["batter"] + bat_feature_cols].dropna()

    bat_scaler = StandardScaler()
    bat_features_scaled = bat_scaler.fit_transform(bat_features[bat_feature_cols])
    bat_features_array = pd.DataFrame(
        bat_features_scaled, columns=bat_feature_cols, index=bat_features.index
    )
    bat_features_array["batter"] = bat_features["batter"].values

    # === Bowlers ===
    bowl = bowler_stats_df[bowler_stats_df["balls_bowled"] >= min_bowl_balls].copy()
    bowl_feature_cols = ["economy", "wickets", "dot_ball_pct"]
    bowl_features = bowl[["bowler"] + bowl_feature_cols].dropna()

    bowl_scaler = StandardScaler()
    bowl_features_scaled = bowl_scaler.fit_transform(bowl_features[bowl_feature_cols])
    bowl_features_array = pd.DataFrame(
        bowl_features_scaled, columns=bowl_feature_cols, index=bowl_features.index
    )
    bowl_features_array["bowler"] = bowl_features["bowler"].values

    return bat_features_array, bowl_features_array, bat_scaler, bowl_scaler


# ===================================================================
# Model 1: Logistic Regression (Baseline Win Predictor)
# ===================================================================

def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train Logistic Regression baseline win predictor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"  Logistic Regression — Accuracy: {metrics['accuracy']}, "
          f"F1: {metrics['f1_score']}, AUC: {metrics['auc_roc']}")

    return {"model": model, "metrics": metrics, "type": "logistic_regression"}


# ===================================================================
# Model 2: Random Forest (Ball-by-Ball Win Probability)
# ===================================================================

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train Random Forest classifier for real-time win probability."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"  Random Forest — Accuracy: {metrics['accuracy']}, "
          f"F1: {metrics['f1_score']}, AUC: {metrics['auc_roc']}")

    return {"model": model, "metrics": metrics, "type": "random_forest"}


# ===================================================================
# Model 3: XGBoost (Primary Win Probability)
# ===================================================================

def train_xgboost(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train XGBoost classifier — primary win probability model."""
    if not HAS_XGBOOST:
        print("  XGBoost skipped (not installed).")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"  XGBoost — Accuracy: {metrics['accuracy']}, "
          f"F1: {metrics['f1_score']}, AUC: {metrics['auc_roc']}")

    return {"model": model, "metrics": metrics, "type": "xgboost"}


# ===================================================================
# Model 4: K-Means Clustering (Player Segmentation)
# ===================================================================

BATSMAN_CLUSTER_LABELS = {
    "high_sr_high_boundary": "Power Hitter",
    "high_avg_low_sr": "Anchor",
    "moderate_all": "All-Rounder",
    "high_sr_low_avg": "Finisher",
}

BOWLER_CLUSTER_LABELS = {
    "low_economy": "Economical",
    "high_wickets": "Wicket-Taker",
    "high_dots": "Death Specialist",
    "moderate_all": "All-Rounder",
}


def train_kmeans_batsmen(bat_features: pd.DataFrame, n_clusters: int = 4) -> dict:
    """Cluster batsmen into performance types."""
    feature_cols = ["strike_rate", "average", "boundary_pct"]
    X = bat_features[feature_cols].values
    names = bat_features["batter"].values

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    sil_score = round(silhouette_score(X, labels), 4) if len(set(labels)) > 1 else 0.0

    # Assign meaningful labels based on cluster centroids
    centroids = model.cluster_centers_
    cluster_names = {}
    for i in range(n_clusters):
        sr, avg, bp = centroids[i]
        if sr > 0.3 and bp > 0.3:
            cluster_names[i] = "Power Hitter"
        elif avg > 0.3 and sr < 0:
            cluster_names[i] = "Anchor"
        elif sr > 0.3 and avg < 0:
            cluster_names[i] = "Finisher"
        else:
            cluster_names[i] = "All-Rounder"

    # Deduplicate names
    used = set()
    fallbacks = ["Power Hitter", "Anchor", "Finisher", "All-Rounder"]
    for i in range(n_clusters):
        if cluster_names[i] in used:
            for fb in fallbacks:
                if fb not in used:
                    cluster_names[i] = fb
                    break
        used.add(cluster_names[i])

    assignments = pd.DataFrame({
        "player": names,
        "cluster": labels,
        "cluster_label": [cluster_names[l] for l in labels],
    })

    print(f"  K-Means Batsmen — Silhouette: {sil_score}, Clusters: {n_clusters}")
    for i in range(n_clusters):
        count = (labels == i).sum()
        print(f"    Cluster {i} ({cluster_names[i]}): {count} players")

    return {
        "model": model,
        "assignments": assignments,
        "cluster_names": cluster_names,
        "silhouette_score": sil_score,
        "type": "kmeans_batsmen",
    }


def train_kmeans_bowlers(bowl_features: pd.DataFrame, n_clusters: int = 4) -> dict:
    """Cluster bowlers into performance types."""
    feature_cols = ["economy", "wickets", "dot_ball_pct"]
    X = bowl_features[feature_cols].values
    names = bowl_features["bowler"].values

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    sil_score = round(silhouette_score(X, labels), 4) if len(set(labels)) > 1 else 0.0

    centroids = model.cluster_centers_
    cluster_names = {}
    for i in range(n_clusters):
        eco, wkt, dots = centroids[i]
        if eco < -0.3:
            cluster_names[i] = "Economical"
        elif wkt > 0.3:
            cluster_names[i] = "Wicket-Taker"
        elif dots > 0.3:
            cluster_names[i] = "Death Specialist"
        else:
            cluster_names[i] = "All-Rounder"

    used = set()
    fallbacks = ["Economical", "Wicket-Taker", "Death Specialist", "All-Rounder"]
    for i in range(n_clusters):
        if cluster_names[i] in used:
            for fb in fallbacks:
                if fb not in used:
                    cluster_names[i] = fb
                    break
        used.add(cluster_names[i])

    assignments = pd.DataFrame({
        "player": names,
        "cluster": labels,
        "cluster_label": [cluster_names[l] for l in labels],
    })

    print(f"  K-Means Bowlers — Silhouette: {sil_score}, Clusters: {n_clusters}")
    for i in range(n_clusters):
        count = (labels == i).sum()
        print(f"    Cluster {i} ({cluster_names[i]}): {count} players")

    return {
        "model": model,
        "assignments": assignments,
        "cluster_names": cluster_names,
        "silhouette_score": sil_score,
        "type": "kmeans_bowlers",
    }


# ===================================================================
# Prediction Helpers
# ===================================================================

def predict_win_probability(
    models: dict,
    target: int,
    current_score: int,
    wickets_fallen: int,
    overs_completed: float,
) -> dict:
    """Predict win probability for the chasing team using all models.

    Parameters
    ----------
    models : dict
        Dictionary with 'random_forest' and 'xgboost' model dicts.
    target : int
        Target score to chase.
    current_score : int
        Runs scored so far in the 2nd innings.
    wickets_fallen : int
        Wickets lost so far.
    overs_completed : float
        Overs bowled so far in the 2nd innings.

    Returns
    -------
    dict
        Win probabilities from each model.
    """
    balls_bowled = int(overs_completed * 6)
    balls_remaining = max(120 - balls_bowled, 0)
    runs_remaining = max(target - current_score, 0)
    current_rr = current_score / overs_completed if overs_completed > 0 else 0.0
    overs_rem = balls_remaining / 6.0
    required_rr = runs_remaining / overs_rem if overs_rem > 0 else 999.0
    required_rr = min(required_rr, 36.0)

    # Match phase
    over_num = int(overs_completed)
    if over_num <= 6:
        phase = 0  # powerplay
    elif over_num <= 15:
        phase = 1  # middle
    else:
        phase = 2  # death

    features = pd.DataFrame([{
        "cum_runs": current_score,
        "cum_wickets": wickets_fallen,
        "overs_completed": overs_completed,
        "current_run_rate": round(current_rr, 2),
        "target": target,
        "runs_remaining": runs_remaining,
        "balls_remaining": balls_remaining,
        "required_run_rate": round(required_rr, 2),
        "match_phase_enc": phase,
    }])

    results = {}

    # Random Forest
    rf = models.get("random_forest")
    if rf and rf.get("model"):
        proba = rf["model"].predict_proba(features)[0]
        results["random_forest"] = {
            "chasing_win_prob": round(float(proba[1]) * 100, 1),
            "defending_win_prob": round(float(proba[0]) * 100, 1),
        }

    # XGBoost
    xgb = models.get("xgboost")
    if xgb and xgb.get("model"):
        proba = xgb["model"].predict_proba(features)[0]
        results["xgboost"] = {
            "chasing_win_prob": round(float(proba[1]) * 100, 1),
            "defending_win_prob": round(float(proba[0]) * 100, 1),
        }

    # Derived stats
    results["match_state"] = {
        "current_run_rate": round(current_rr, 2),
        "required_run_rate": round(required_rr, 2),
        "runs_remaining": runs_remaining,
        "balls_remaining": balls_remaining,
        "match_phase": ["Powerplay", "Middle Overs", "Death Overs"][phase],
    }

    return results


# ===================================================================
# Persistence
# ===================================================================

def save_all_models(models: dict, base_dir: str = "models") -> None:
    """Save all trained models to disk."""
    os.makedirs(base_dir, exist_ok=True)

    for name, model_dict in models.items():
        if model_dict is not None:
            path = os.path.join(base_dir, f"{name}.joblib")
            joblib.dump(model_dict, path)
            print(f"  Saved: {path}")


def load_all_models(base_dir: str = "models") -> dict:
    """Load all trained models from disk."""
    models = {}
    model_names = [
        "logistic_regression", "random_forest", "xgboost",
        "kmeans_batsmen", "kmeans_bowlers",
    ]

    for name in model_names:
        path = os.path.join(base_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            models[name] = None

    return models
