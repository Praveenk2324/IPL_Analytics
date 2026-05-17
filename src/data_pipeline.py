"""
IPL Analytics — Data Ingestion, Cleaning & Feature Engineering Pipeline.

Loads raw IPL match and delivery CSVs, standardizes team names,
handles missing values, engineers features for ML models, and
provides cleaned DataFrames for analysis and prediction.
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Team-name standardization mapping
# ---------------------------------------------------------------------------
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

# Venue → City fallback mapping (for null cities)
VENUE_CITY_MAP = {
    "Dubai International Cricket Stadium": "Dubai",
    "Sheikh Zayed Stadium": "Abu Dhabi",
    "Sharjah Cricket Stadium": "Sharjah",
    "Newlands": "Cape Town",
    "St George's Park": "Port Elizabeth",
    "Kingsmead": "Durban",
    "SuperSport Park": "Centurion",
    "Buffalo Park": "East London",
    "De Beers Diamond Oval": "Kimberley",
    "OUTsurance Oval": "Bloemfontein",
    "Brabourne Stadium": "Mumbai",
    "Wankhede Stadium": "Mumbai",
    "M Chinnaswamy Stadium": "Bangalore",
    "MA Chidambaram Stadium": "Chennai",
    "Rajiv Gandhi International Stadium": "Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal": "Hyderabad",
    "Eden Gardens": "Kolkata",
    "Feroz Shah Kotla": "Delhi",
    "Arun Jaitley Stadium": "Delhi",
    "Arun Jaitley Stadium, Delhi": "Delhi",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Chandigarh",
    "Punjab Cricket Association Stadium, Mohali": "Chandigarh",
    "IS Bindra Stadium": "Chandigarh",
    "Sawai Mansingh Stadium": "Jaipur",
    "DY Patil Stadium": "Navi Mumbai",
    "Narendra Modi Stadium, Ahmedabad": "Ahmedabad",
    "Zayed Cricket Stadium, Abu Dhabi": "Abu Dhabi",
    "Maharashtra Cricket Association Stadium": "Pune",
    "Maharashtra Cricket Association Stadium, Pune": "Pune",
    "Dr DY Patil Sports Academy": "Navi Mumbai",
    "Dr DY Patil Sports Academy, Mumbai": "Navi Mumbai",
    "Himachal Pradesh Cricket Association Stadium": "Dharamsala",
    "Barsapara Cricket Stadium, Guwahati": "Guwahati",
}


# ===================================================================
# Data Loading
# ===================================================================

def load_raw_data(
    matches_path: str = "data/raw/2008_2016/matches.csv",
    deliveries_path: str = "data/raw/2008_2016/deliveries.csv",
) -> tuple:
    """Load match and delivery CSVs with robust error handling.

    Parameters
    ----------
    matches_path : str
        File path to the matches CSV.
    deliveries_path : str
        File path to the deliveries CSV.

    Returns
    -------
    tuple
        (matches_df, deliveries_df).
    """
    for path, label in [(matches_path, "Matches"), (deliveries_path, "Deliveries")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} CSV not found at '{path}'. "
                "Please place it in the data/ directory."
            )

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries


# ===================================================================
# Cleaning
# ===================================================================

def standardize_team_names(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Replace legacy team names with current franchise names."""
    if columns is None:
        candidates = [
            "team1", "team2", "toss_winner", "winner",
            "batting_team", "bowling_team",
        ]
        columns = [c for c in candidates if c in df.columns]

    df = df.copy()
    for col in columns:
        df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def handle_missing_values(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the matches DataFrame."""
    matches = matches.copy()

    if "city" in matches.columns and "venue" in matches.columns:
        mask = matches["city"].isnull()
        matches.loc[mask, "city"] = matches.loc[mask, "venue"].map(VENUE_CITY_MAP)
        matches["city"] = matches["city"].fillna("Unknown")

    matches["winner"] = matches["winner"].fillna("No Result")
    matches["player_of_match"] = matches["player_of_match"].fillna("Not Awarded")
    matches["result_margin"] = matches["result_margin"].fillna(0)
    matches["method"] = matches["method"].fillna("Normal")

    return matches


# ===================================================================
# Feature Engineering — Match Level
# ===================================================================

def engineer_match_features(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Add match-level features for ML models.

    Adds:
      - toss_win_is_match_win: bool
      - first_innings_score: total runs in first innings
      - second_innings_score: total runs in second innings
      - is_chasing_team_winner: did the team batting second win?
      - batting_first_team / batting_second_team
    """
    matches = matches.copy()

    # Toss → match win flag
    matches["toss_win_is_match_win"] = (
        matches["toss_winner"] == matches["winner"]
    ).astype(int)

    # First innings score per match
    first_inn = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "first_innings_score"})
    )

    # Second innings score per match
    second_inn = (
        deliveries[deliveries["inning"] == 2]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "second_innings_score"})
    )

    # Batting first / second teams
    batting_first = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["batting_team"]
        .first()
        .reset_index()
        .rename(columns={"batting_team": "batting_first_team"})
    )
    batting_second = (
        deliveries[deliveries["inning"] == 2]
        .groupby("match_id")["batting_team"]
        .first()
        .reset_index()
        .rename(columns={"batting_team": "batting_second_team"})
    )

    matches = matches.merge(first_inn, left_on="id", right_on="match_id", how="left")
    matches.drop(columns=["match_id"], inplace=True, errors="ignore")

    matches = matches.merge(second_inn, left_on="id", right_on="match_id", how="left")
    matches.drop(columns=["match_id"], inplace=True, errors="ignore")

    matches = matches.merge(batting_first, left_on="id", right_on="match_id", how="left")
    matches.drop(columns=["match_id"], inplace=True, errors="ignore")

    matches = matches.merge(batting_second, left_on="id", right_on="match_id", how="left")
    matches.drop(columns=["match_id"], inplace=True, errors="ignore")

    matches["first_innings_score"] = matches["first_innings_score"].fillna(0).astype(int)
    matches["second_innings_score"] = matches["second_innings_score"].fillna(0).astype(int)

    # Did chasing team win?
    matches["is_chasing_team_winner"] = (
        matches["winner"] == matches["batting_second_team"]
    ).astype(int)

    return matches


# ===================================================================
# Feature Engineering — Ball Level (for win probability models)
# ===================================================================

def engineer_ball_features(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Add ball-level features for real-time win probability.

    Adds (for 2nd innings deliveries):
      - cum_runs: cumulative runs scored so far
      - cum_wickets: cumulative wickets lost
      - balls_bowled: balls bowled in this innings
      - overs_completed: balls_bowled / 6
      - current_run_rate: cum_runs / overs_completed
      - target: target score (first innings + 1)
      - runs_remaining: target - cum_runs
      - balls_remaining: 120 - balls_bowled
      - required_run_rate: runs_remaining / (balls_remaining / 6)
      - match_phase: powerplay / middle / death
      - batting_team_won: binary target label
    """
    # Merge match winner and target info
    match_info = matches[["id", "winner", "target_runs"]].copy()
    match_info.rename(columns={"id": "match_id"}, inplace=True)

    # First innings totals for target calculation
    first_inn_totals = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "first_innings_total"})
    )

    # Work with second innings only
    second = deliveries[deliveries["inning"] == 2].copy()
    second = second.sort_values(["match_id", "over", "ball"])

    second = second.merge(match_info, on="match_id", how="left")
    second = second.merge(first_inn_totals, on="match_id", how="left")

    # Target = first innings total + 1
    second["target"] = second["first_innings_total"] + 1

    # Cumulative runs and wickets
    second["cum_runs"] = second.groupby("match_id")["total_runs"].cumsum()
    second["cum_wickets"] = second.groupby("match_id")["is_wicket"].cumsum()

    # Balls bowled (count within innings)
    second["balls_bowled"] = second.groupby("match_id").cumcount() + 1

    # Overs completed
    second["overs_completed"] = second["balls_bowled"] / 6.0

    # Current run rate
    second["current_run_rate"] = np.where(
        second["overs_completed"] > 0,
        second["cum_runs"] / second["overs_completed"],
        0.0,
    )

    # Runs and balls remaining
    second["runs_remaining"] = second["target"] - second["cum_runs"]
    second["balls_remaining"] = 120 - second["balls_bowled"]
    second["balls_remaining"] = second["balls_remaining"].clip(lower=0)

    # Required run rate
    overs_rem = second["balls_remaining"] / 6.0
    second["required_run_rate"] = np.where(
        overs_rem > 0,
        second["runs_remaining"] / overs_rem,
        999.0,  # innings over
    )

    # Match phase
    second["match_phase"] = pd.cut(
        second["over"],
        bins=[-1, 5, 15, 20],
        labels=["powerplay", "middle", "death"],
    )

    # Target: did the batting team (chasing) win?
    second["batting_team_won"] = (
        second["batting_team"] == second["winner"]
    ).astype(int)

    return second


# ===================================================================
# Merge Utility
# ===================================================================

def merge_data(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Merge ball-by-ball deliveries with match metadata."""
    merge_cols = ["id", "season", "city", "venue", "date"]
    available = [c for c in merge_cols if c in matches.columns]
    merged = deliveries.merge(
        matches[available],
        left_on="match_id",
        right_on="id",
        how="left",
    )
    if "id" in merged.columns:
        merged.drop(columns=["id"], inplace=True)
    return merged


# ===================================================================
# Analysis Helpers (for API endpoints)
# ===================================================================

def batsman_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate batting statistics for every batsman."""
    runs = (
        deliveries.groupby("batter")["batsman_runs"]
        .sum()
        .reset_index()
        .rename(columns={"batsman_runs": "runs"})
    )

    valid_balls = deliveries[deliveries["extras_type"].fillna("none") != "wides"]
    balls = valid_balls.groupby("batter").size().reset_index(name="balls_faced")

    dismissed = deliveries[deliveries["is_wicket"] == 1]
    dismissals = (
        dismissed.groupby("player_dismissed")
        .size()
        .reset_index(name="dismissals")
        .rename(columns={"player_dismissed": "batter"})
    )

    stats = runs.merge(balls, on="batter", how="left")
    stats = stats.merge(dismissals, on="batter", how="left")
    stats["dismissals"] = stats["dismissals"].fillna(0).astype(int)
    stats["balls_faced"] = stats["balls_faced"].fillna(0).astype(int)

    stats["strike_rate"] = np.where(
        stats["balls_faced"] > 0,
        (stats["runs"] / stats["balls_faced"]) * 100,
        0.0,
    )
    stats["average"] = np.where(
        stats["dismissals"] > 0,
        stats["runs"] / stats["dismissals"],
        stats["runs"].astype(float),
    )
    stats["strike_rate"] = stats["strike_rate"].round(2)
    stats["average"] = stats["average"].round(2)

    # Boundary stats
    fours = deliveries[deliveries["batsman_runs"] == 4].groupby("batter").size().reset_index(name="fours")
    sixes = deliveries[deliveries["batsman_runs"] == 6].groupby("batter").size().reset_index(name="sixes")
    stats = stats.merge(fours, on="batter", how="left")
    stats = stats.merge(sixes, on="batter", how="left")
    stats["fours"] = stats["fours"].fillna(0).astype(int)
    stats["sixes"] = stats["sixes"].fillna(0).astype(int)
    stats["boundary_pct"] = np.where(
        stats["balls_faced"] > 0,
        ((stats["fours"] + stats["sixes"]) / stats["balls_faced"]) * 100,
        0.0,
    ).round(2)

    return stats.sort_values("runs", ascending=False).reset_index(drop=True)


def bowler_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate bowling statistics for every bowler."""
    df = deliveries.copy()
    bowler_extras = df["extras_type"].isin(["wides", "noballs"])
    df["runs_conceded"] = df["batsman_runs"] + np.where(
        bowler_extras, df["extra_runs"], 0
    )

    runs = df.groupby("bowler")["runs_conceded"].sum().reset_index()

    legal = df[~df["extras_type"].isin(["wides", "noballs"])]
    balls = legal.groupby("bowler").size().reset_index(name="balls_bowled")

    # Dot balls
    dots = legal[legal["total_runs"] == 0].groupby("bowler").size().reset_index(name="dot_balls")

    wicket_types_excluded = ["run out", "retired hurt", "obstructing the field"]
    wickets_df = df[
        (df["is_wicket"] == 1)
        & (~df["dismissal_kind"].isin(wicket_types_excluded))
    ]
    wickets = wickets_df.groupby("bowler").size().reset_index(name="wickets")

    stats = runs.merge(balls, on="bowler", how="left")
    stats = stats.merge(wickets, on="bowler", how="left")
    stats = stats.merge(dots, on="bowler", how="left")
    stats["wickets"] = stats["wickets"].fillna(0).astype(int)
    stats["balls_bowled"] = stats["balls_bowled"].fillna(0).astype(int)
    stats["dot_balls"] = stats["dot_balls"].fillna(0).astype(int)
    stats["overs"] = (stats["balls_bowled"] / 6).round(2)
    stats["economy"] = np.where(
        stats["overs"] > 0,
        (stats["runs_conceded"] / stats["overs"]).round(2),
        0.0,
    )
    stats["dot_ball_pct"] = np.where(
        stats["balls_bowled"] > 0,
        (stats["dot_balls"] / stats["balls_bowled"] * 100).round(2),
        0.0,
    )

    return stats.sort_values("wickets", ascending=False).reset_index(drop=True)


def batsman_season_runs(merged: pd.DataFrame, player: str) -> pd.DataFrame:
    """Season-wise run tally for a specific batsman."""
    player_data = merged[merged["batter"] == player]
    return (
        player_data.groupby("season")["batsman_runs"]
        .sum()
        .reset_index()
        .rename(columns={"batsman_runs": "runs"})
    )


def bowler_season_wickets(merged: pd.DataFrame, player: str) -> pd.DataFrame:
    """Season-wise wicket tally for a specific bowler."""
    wicket_types_excluded = ["run out", "retired hurt", "obstructing the field"]
    player_data = merged[
        (merged["bowler"] == player)
        & (merged["is_wicket"] == 1)
        & (~merged["dismissal_kind"].isin(wicket_types_excluded))
    ]
    return player_data.groupby("season").size().reset_index(name="wickets")


def venue_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Venue analytics: avg first innings score, chase vs defend win rates."""
    # Merge for first innings scores
    first_inn = deliveries[deliveries["inning"] == 1]
    match_totals = (
        first_inn.groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "first_innings_score"})
    )

    venue_matches = matches[["id", "venue", "winner"]].merge(
        match_totals, left_on="id", right_on="match_id", how="inner"
    )
    venue_matches.drop(columns=["match_id"], inplace=True, errors="ignore")

    # Batting first team per match
    bat_first = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["batting_team"]
        .first()
        .reset_index()
        .rename(columns={"batting_team": "batting_first"})
    )
    venue_matches = venue_matches.merge(bat_first, left_on="id", right_on="match_id", how="left")
    venue_matches.drop(columns=["match_id"], inplace=True, errors="ignore")

    venue_matches["defend_win"] = (venue_matches["winner"] == venue_matches["batting_first"]).astype(int)
    venue_matches["chase_win"] = 1 - venue_matches["defend_win"]
    # Handle no-result
    no_result = venue_matches["winner"] == "No Result"
    venue_matches.loc[no_result, "defend_win"] = 0
    venue_matches.loc[no_result, "chase_win"] = 0

    result = venue_matches.groupby("venue").agg(
        avg_score=("first_innings_score", "mean"),
        total_matches=("venue", "size"),
        defend_wins=("defend_win", "sum"),
        chase_wins=("chase_win", "sum"),
    ).reset_index()

    result["avg_score"] = result["avg_score"].round(1)
    result["defend_win_pct"] = (result["defend_wins"] / result["total_matches"] * 100).round(1)
    result["chase_win_pct"] = (result["chase_wins"] / result["total_matches"] * 100).round(1)

    return result.sort_values("avg_score", ascending=False).reset_index(drop=True)


def team_win_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Win count and win percentage for each team."""
    valid = matches[matches["winner"] != "No Result"]
    team1_counts = valid["team1"].value_counts()
    team2_counts = valid["team2"].value_counts()
    played = (team1_counts.add(team2_counts, fill_value=0)).astype(int)
    wins = valid["winner"].value_counts()

    stats = pd.DataFrame({"matches_played": played, "wins": wins}).fillna(0)
    stats["wins"] = stats["wins"].astype(int)
    stats["win_pct"] = ((stats["wins"] / stats["matches_played"]) * 100).round(1)
    stats = stats.reset_index().rename(columns={"index": "team"})
    return stats.sort_values("wins", ascending=False).reset_index(drop=True)


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """Head-to-head record between two teams."""
    h2h = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ]

    return {
        "total": int(len(h2h)),
        "team_a_wins": int((h2h["winner"] == team_a).sum()),
        "team_b_wins": int((h2h["winner"] == team_b).sum()),
        "no_result": int((h2h["winner"] == "No Result").sum()),
    }


def head_to_head_top_performers(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    team_a: str,
    team_b: str,
    top_n: int = 5,
) -> dict:
    """Top batsmen and bowlers in head-to-head encounters."""
    h2h_matches = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ]
    h2h_ids = h2h_matches["id"].tolist()
    h2h_del = deliveries[deliveries["match_id"].isin(h2h_ids)]

    top_batsmen = (
        h2h_del.groupby("batter")["batsman_runs"]
        .sum()
        .nlargest(top_n)
        .reset_index()
        .rename(columns={"batsman_runs": "runs"})
    )

    wicket_types_excluded = ["run out", "retired hurt", "obstructing the field"]
    wickets = h2h_del[
        (h2h_del["is_wicket"] == 1)
        & (~h2h_del["dismissal_kind"].isin(wicket_types_excluded))
    ]
    top_bowlers = (
        wickets.groupby("bowler")
        .size()
        .nlargest(top_n)
        .reset_index(name="wickets")
    )

    return {
        "top_batsmen": top_batsmen.to_dict(orient="records"),
        "top_bowlers": top_bowlers.to_dict(orient="records"),
    }


def season_performance(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Season-wise win record for a specific team."""
    team_matches = matches[
        (matches["team1"] == team) | (matches["team2"] == team)
    ].copy()
    played = team_matches.groupby("season").size().reset_index(name="played")
    wins = (
        team_matches[team_matches["winner"] == team]
        .groupby("season").size().reset_index(name="wins")
    )
    result = played.merge(wins, on="season", how="left")
    result["wins"] = result["wins"].fillna(0).astype(int)
    result["win_pct"] = ((result["wins"] / result["played"]) * 100).round(1)
    return result


def get_all_teams(matches: pd.DataFrame) -> list:
    """Return sorted list of all unique team names."""
    teams = set(matches["team1"].unique()) | set(matches["team2"].unique())
    return sorted([t for t in teams if t != "No Result"])


def get_all_batsmen(deliveries: pd.DataFrame, min_balls: int = 100) -> list:
    """Return sorted list of batsmen with at least min_balls faced."""
    counts = deliveries.groupby("batter").size()
    return sorted(counts[counts >= min_balls].index.tolist())


def get_all_bowlers(deliveries: pd.DataFrame, min_balls: int = 100) -> list:
    """Return sorted list of bowlers with at least min_balls bowled."""
    counts = deliveries.groupby("bowler").size()
    return sorted(counts[counts >= min_balls].index.tolist())


def total_sixes_fours(deliveries: pd.DataFrame) -> dict:
    """Total 4s and 6s across all IPL matches."""
    return {
        "fours": int((deliveries["batsman_runs"] == 4).sum()),
        "sixes": int((deliveries["batsman_runs"] == 6).sum()),
    }


def matches_per_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Count of matches per season."""
    return (
        matches.groupby("season")
        .size()
        .reset_index(name="matches")
        .sort_values("season")
    )


# ===================================================================
# Orchestrator
# ===================================================================

def get_clean_data(
    matches_path: str = "data/raw/2008_2016/matches.csv",
    deliveries_path: str = "data/raw/2008_2016/deliveries.csv",
) -> tuple:
    """One-call orchestrator: load → clean → engineer → merge.

    Returns
    -------
    tuple
        (matches_clean, deliveries_clean, merged).
    """
    matches, deliveries = load_raw_data(matches_path, deliveries_path)

    matches = standardize_team_names(matches)
    matches = handle_missing_values(matches)
    deliveries = standardize_team_names(deliveries)

    matches = engineer_match_features(matches, deliveries)
    merged = merge_data(matches, deliveries)

    return matches, deliveries, merged
