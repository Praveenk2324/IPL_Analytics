"""
IPL Analytics — Model Training Script.

Runs the full training pipeline:
1. Load and clean data
2. Engineer features
3. Train all 4 models (LogReg, RF, XGBoost, K-Means)
4. Save models to models/
"""

import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import (
    get_clean_data,
    engineer_ball_features,
    batsman_stats,
    bowler_stats,
)
from src.ml_models import (
    prepare_match_level_features,
    prepare_ball_level_features,
    prepare_player_clustering_features,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_kmeans_batsmen,
    train_kmeans_bowlers,
    save_all_models,
)


def main():
    print("=" * 60)
    print("IPL ANALYTICS — MODEL TRAINING PIPELINE")
    print("=" * 60)

    start = time.time()

    # ── Step 1: Load & Clean Data ────────────────────────────────
    print("\n[1/5] Loading and cleaning data...")
    matches, deliveries, merged = get_clean_data()
    print(f"  Matches: {len(matches)}, Deliveries: {len(deliveries):,}")

    # ── Step 2: Train Logistic Regression ────────────────────────
    print("\n[2/5] Training Logistic Regression (match-level baseline)...")
    X_match, y_match, match_encoders = prepare_match_level_features(matches)
    lr_result = train_logistic_regression(X_match, y_match)
    lr_result["encoders"] = match_encoders

    # ── Step 3: Prepare ball-level features & train RF + XGBoost ─
    print("\n[3/5] Engineering ball-level features...")
    ball_features = engineer_ball_features(deliveries, matches)
    X_ball, y_ball = prepare_ball_level_features(ball_features)
    print(f"  Ball-level samples: {len(X_ball):,}")

    print("\n  Training Random Forest...")
    rf_result = train_random_forest(X_ball, y_ball)

    print("\n  Training XGBoost...")
    xgb_result = train_xgboost(X_ball, y_ball)

    # ── Step 4: K-Means Player Clustering ────────────────────────
    print("\n[4/5] Training K-Means player clustering...")
    bat_stats = batsman_stats(deliveries)
    bowl_stats_df = bowler_stats(deliveries)

    bat_features, bowl_features, bat_scaler, bowl_scaler = (
        prepare_player_clustering_features(bat_stats, bowl_stats_df)
    )

    bat_km = train_kmeans_batsmen(bat_features)
    bowl_km = train_kmeans_bowlers(bowl_features)

    # ── Step 5: Save all models ──────────────────────────────────
    print("\n[5/5] Saving models...")
    all_models = {
        "logistic_regression": lr_result,
        "random_forest": rf_result,
        "xgboost": xgb_result,
        "kmeans_batsmen": bat_km,
        "kmeans_bowlers": bowl_km,
    }
    save_all_models(all_models)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Print summary
    print("\n[*] Model Performance Summary:")
    print("-" * 40)
    for name, result in all_models.items():
        if result is None:
            print(f"  {name}: SKIPPED")
            continue
        if "metrics" in result:
            m = result["metrics"]
            print(f"  {name}:")
            for k, v in m.items():
                print(f"    {k}: {v}")
        elif "silhouette_score" in result:
            print(f"  {name}: silhouette={result['silhouette_score']}")
    print()


if __name__ == "__main__":
    main()
