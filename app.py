"""
IPL Sports Analytics — Flask Application.

Serves the interactive dashboard and provides REST API endpoints
for win probability prediction, player stats, venue analytics,
and head-to-head comparisons.
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np


def _convert_types(d: dict) -> dict:
    """Convert numpy/pandas types to native Python for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        elif isinstance(v, (np.ndarray,)):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out

from src.data_pipeline import (
    get_clean_data,
    batsman_stats,
    bowler_stats,
    batsman_season_runs,
    bowler_season_wickets,
    venue_stats,
    team_win_stats,
    head_to_head,
    head_to_head_top_performers,
    season_performance,
    get_all_teams,
    get_all_batsmen,
    get_all_bowlers,
    total_sixes_fours,
    matches_per_season,
)
from src.ml_models import load_all_models, predict_win_probability


# ── App Setup ────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load Data & Models (once at startup) ─────────────────────────
print("Loading IPL data...")
try:
    MATCHES, DELIVERIES, MERGED = get_clean_data()
    print(f"  Data loaded: {len(MATCHES)} matches, {len(DELIVERIES):,} deliveries")
except Exception as e:
    print(f"ERROR loading data: {e}")
    MATCHES, DELIVERIES, MERGED = None, None, None

print("Loading ML models...")
MODELS = load_all_models()
loaded = [k for k, v in MODELS.items() if v is not None]
print(f"  Models loaded: {loaded}")

# Pre-compute expensive stats
print("Pre-computing stats...")
BAT_STATS = batsman_stats(DELIVERIES) if DELIVERIES is not None else pd.DataFrame()
BOWL_STATS = bowler_stats(DELIVERIES) if DELIVERIES is not None else pd.DataFrame()
print("  Stats ready.")


# ── Page Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SPA dashboard."""
    return render_template("index.html")


# ── API: Overview / KPIs ─────────────────────────────────────────

@app.route("/api/overview")
def api_overview():
    """Return high-level KPI stats."""
    boundaries = total_sixes_fours(DELIVERIES)
    mps = matches_per_season(MATCHES).to_dict(orient="records")

    # Top run scorers
    top_runs = (
        DELIVERIES.groupby("batter")["batsman_runs"]
        .sum()
        .nlargest(10)
        .reset_index()
        .rename(columns={"batsman_runs": "runs"})
        .to_dict(orient="records")
    )

    return jsonify({
        "total_matches": int(len(MATCHES)),
        "total_seasons": int(MATCHES["season"].nunique()),
        "total_sixes": boundaries["sixes"],
        "total_fours": boundaries["fours"],
        "matches_per_season": mps,
        "top_run_scorers": top_runs,
    })


# ── API: Teams ───────────────────────────────────────────────────

@app.route("/api/teams")
def api_teams():
    """Return list of all teams."""
    teams = get_all_teams(MATCHES)
    return jsonify({"teams": teams})


@app.route("/api/team-stats")
def api_team_stats():
    """Return win stats for all teams."""
    stats = team_win_stats(MATCHES).to_dict(orient="records")
    return jsonify({"stats": stats})


@app.route("/api/team-season/<team>")
def api_team_season(team):
    """Return season-wise performance for a team."""
    perf = season_performance(MATCHES, team).to_dict(orient="records")
    return jsonify({"team": team, "seasons": perf})


# ── API: Players ─────────────────────────────────────────────────

@app.route("/api/players")
def api_players():
    """Return lists of batsmen and bowlers."""
    batsmen = get_all_batsmen(DELIVERIES, min_balls=100)
    bowlers = get_all_bowlers(DELIVERIES, min_balls=100)
    return jsonify({"batsmen": batsmen, "bowlers": bowlers})


@app.route("/api/player-stats/batsman/<name>")
def api_batsman_stats(name):
    """Return batting stats for a specific player."""
    row = BAT_STATS[BAT_STATS["batter"] == name]
    if row.empty:
        return jsonify({"error": "Player not found"}), 404

    stats = _convert_types(row.iloc[0].to_dict())

    season = batsman_season_runs(MERGED, name).to_dict(orient="records")

    return jsonify({"stats": stats, "season_runs": season})


@app.route("/api/player-stats/bowler/<name>")
def api_bowler_stats(name):
    """Return bowling stats for a specific player."""
    row = BOWL_STATS[BOWL_STATS["bowler"] == name]
    if row.empty:
        return jsonify({"error": "Player not found"}), 404

    stats = _convert_types(row.iloc[0].to_dict())

    season = bowler_season_wickets(MERGED, name).to_dict(orient="records")

    return jsonify({"stats": stats, "season_wickets": season})


@app.route("/api/player-stats/top-batsmen")
def api_top_batsmen():
    """Return top batsmen by various metrics."""
    metric = request.args.get("metric", "runs")
    n = int(request.args.get("n", 10))
    min_balls = int(request.args.get("min_balls", 500))

    qualified = BAT_STATS[BAT_STATS["balls_faced"] >= min_balls]
    if metric == "strike_rate":
        top = qualified.nlargest(n, "strike_rate")
    elif metric == "average":
        top = qualified.nlargest(n, "average")
    else:
        top = qualified.nlargest(n, "runs")

    return jsonify({"players": top.to_dict(orient="records")})


@app.route("/api/player-stats/top-bowlers")
def api_top_bowlers():
    """Return top bowlers by various metrics."""
    metric = request.args.get("metric", "wickets")
    n = int(request.args.get("n", 10))
    min_balls = int(request.args.get("min_balls", 300))

    qualified = BOWL_STATS[BOWL_STATS["balls_bowled"] >= min_balls]
    if metric == "economy":
        top = qualified.nsmallest(n, "economy")
    else:
        top = qualified.nlargest(n, metric)

    return jsonify({"players": top.to_dict(orient="records")})


# ── API: Player Clusters ─────────────────────────────────────────

@app.route("/api/player-clusters")
def api_player_clusters():
    """Return K-Means cluster assignments."""
    result = {}

    bat_km = MODELS.get("kmeans_batsmen")
    if bat_km and "assignments" in bat_km:
        result["batsmen"] = {
            "players": bat_km["assignments"].to_dict(orient="records"),
            "cluster_names": bat_km.get("cluster_names", {}),
            "silhouette_score": bat_km.get("silhouette_score", 0),
        }

    bowl_km = MODELS.get("kmeans_bowlers")
    if bowl_km and "assignments" in bowl_km:
        result["bowlers"] = {
            "players": bowl_km["assignments"].to_dict(orient="records"),
            "cluster_names": bowl_km.get("cluster_names", {}),
            "silhouette_score": bowl_km.get("silhouette_score", 0),
        }

    return jsonify(result)


# ── API: Venue Stats ─────────────────────────────────────────────

@app.route("/api/venue-stats")
def api_venue_stats():
    """Return venue analytics with chase/defend win rates."""
    stats = venue_stats(MATCHES, DELIVERIES)
    min_matches = int(request.args.get("min_matches", 5))
    stats = stats[stats["total_matches"] >= min_matches]
    return jsonify({"venues": stats.to_dict(orient="records")})


# ── API: Head-to-Head ────────────────────────────────────────────

@app.route("/api/head-to-head")
def api_head_to_head():
    """Return head-to-head record between two teams."""
    team_a = request.args.get("team_a", "")
    team_b = request.args.get("team_b", "")
    if not team_a or not team_b:
        return jsonify({"error": "team_a and team_b required"}), 400

    record = head_to_head(MATCHES, team_a, team_b)
    performers = head_to_head_top_performers(MATCHES, DELIVERIES, team_a, team_b)

    return jsonify({
        "team_a": team_a,
        "team_b": team_b,
        "record": record,
        "top_performers": performers,
    })


# ── API: Win Probability Prediction ──────────────────────────────

@app.route("/api/predict-win", methods=["POST"])
def api_predict_win():
    """Predict win probability based on current match state.

    Expected JSON body:
    {
        "target": 180,
        "current_score": 85,
        "wickets_fallen": 3,
        "overs_completed": 10.0
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required_fields = ["target", "current_score", "wickets_fallen", "overs_completed"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        result = predict_win_probability(
            models=MODELS,
            target=int(data["target"]),
            current_score=int(data["current_score"]),
            wickets_fallen=int(data["wickets_fallen"]),
            overs_completed=float(data["overs_completed"]),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Model Info ──────────────────────────────────────────────

@app.route("/api/model-info")
def api_model_info():
    """Return information about loaded ML models."""
    info = {}
    for name, model_dict in MODELS.items():
        if model_dict is None:
            info[name] = {"status": "not loaded"}
        else:
            entry = {"status": "loaded", "type": model_dict.get("type", name)}
            if "metrics" in model_dict:
                entry["metrics"] = model_dict["metrics"]
            if "silhouette_score" in model_dict:
                entry["silhouette_score"] = model_dict["silhouette_score"]
            info[name] = entry
    return jsonify(info)


# ── API: Seasons ─────────────────────────────────────────────────

@app.route("/api/seasons")
def api_seasons():
    """Return list of all seasons."""
    seasons = sorted(MATCHES["season"].unique().tolist())
    return jsonify({"seasons": seasons})


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
