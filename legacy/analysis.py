"""
IPL Analytics — Advanced Cricket Metric Engineering.

Provides functions for computing batsman, bowler, venue, and team
analytics from cleaned IPL data.
"""

import pandas as pd
import numpy as np


# ===================================================================
# Batsman Metrics
# ===================================================================

def batsman_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate batting statistics for every batsman.

    Metrics
    -------
    - **Runs** : total runs scored off the bat.
    - **Balls Faced** : legal deliveries faced (excludes wides).
    - **Dismissals** : number of times dismissed.
    - **Strike Rate** : (Runs / Balls Faced) × 100.
    - **Average** : Runs / Dismissals (∞ if never dismissed → capped at runs).

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball deliveries data.

    Returns
    -------
    pd.DataFrame
        One row per batsman with columns
        [batter, runs, balls_faced, dismissals, strike_rate, average].
    """
    # Runs scored by each batsman
    runs = (
        deliveries.groupby("batter")["batsman_runs"]
        .sum()
        .reset_index()
        .rename(columns={"batsman_runs": "runs"})
    )

    # Balls faced: exclude wides (wides are not counted as balls faced)
    valid_balls = deliveries[
        deliveries["extras_type"].fillna("none") != "wides"
    ]
    balls = (
        valid_balls.groupby("batter")
        .size()
        .reset_index(name="balls_faced")
    )

    # Dismissals
    dismissed = deliveries[deliveries["is_wicket"] == 1]
    dismissals = (
        dismissed.groupby("player_dismissed")
        .size()
        .reset_index(name="dismissals")
        .rename(columns={"player_dismissed": "batter"})
    )

    # Merge
    stats = runs.merge(balls, on="batter", how="left")
    stats = stats.merge(dismissals, on="batter", how="left")
    stats["dismissals"] = stats["dismissals"].fillna(0).astype(int)

    # Strike Rate = (Runs / Balls Faced) × 100
    stats["strike_rate"] = np.where(
        stats["balls_faced"] > 0,
        (stats["runs"] / stats["balls_faced"]) * 100,
        0.0,
    )

    # Batting Average = Runs / Dismissals
    stats["average"] = np.where(
        stats["dismissals"] > 0,
        stats["runs"] / stats["dismissals"],
        stats["runs"].astype(float),  # not-out → average = runs
    )

    stats["strike_rate"] = stats["strike_rate"].round(2)
    stats["average"] = stats["average"].round(2)

    return stats.sort_values("runs", ascending=False).reset_index(drop=True)


def top_batsmen(
    deliveries: pd.DataFrame,
    metric: str = "runs",
    n: int = 10,
    min_balls: int = 200,
) -> pd.DataFrame:
    """Return top-N batsmen ranked by a given metric.

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball data.
    metric : str
        One of 'runs', 'strike_rate', 'average'.
    n : int
        Number of batsmen to return.
    min_balls : int
        Minimum balls faced to qualify (avoids inflated rates).

    Returns
    -------
    pd.DataFrame
        Top-N batsmen sorted descending by *metric*.
    """
    stats = batsman_stats(deliveries)
    qualified = stats[stats["balls_faced"] >= min_balls]
    return qualified.nlargest(n, metric).reset_index(drop=True)


def batsman_season_runs(
    merged: pd.DataFrame,
    player: str,
) -> pd.DataFrame:
    """Get season-wise run tally for a specific batsman.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged match + delivery data with 'season' column.
    player : str
        Batsman name.

    Returns
    -------
    pd.DataFrame
        Columns: [season, runs].
    """
    player_data = merged[merged["batter"] == player]
    season_runs = (
        player_data.groupby("season")["batsman_runs"]
        .sum()
        .reset_index()
        .rename(columns={"batsman_runs": "runs"})
    )
    return season_runs


# ===================================================================
# Bowler Metrics
# ===================================================================

def bowler_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate bowling statistics for every bowler.

    Metrics
    -------
    - **Runs Conceded** : total_runs given (excluding leg-byes and byes).
    - **Balls Bowled** : legal deliveries bowled (excludes wides & no-balls).
    - **Overs** : Balls Bowled / 6.
    - **Wickets** : dismissals attributed to the bowler (excludes run-outs).
    - **Economy Rate** : Runs Conceded / Overs.

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball deliveries data.

    Returns
    -------
    pd.DataFrame
        One row per bowler.
    """
    # Runs conceded: batsman_runs + extras charged to bowler (wides, noballs)
    df = deliveries.copy()
    bowler_extras = df["extras_type"].isin(["wides", "noballs"])
    df["runs_conceded"] = df["batsman_runs"] + np.where(
        bowler_extras, df["extra_runs"], 0
    )

    runs = (
        df.groupby("bowler")["runs_conceded"]
        .sum()
        .reset_index()
    )

    # Legal deliveries (exclude wides & no-balls)
    legal = df[~df["extras_type"].isin(["wides", "noballs"])]
    balls = (
        legal.groupby("bowler")
        .size()
        .reset_index(name="balls_bowled")
    )

    # Wickets (exclude run-outs, retired hurt, obstructing the field)
    wicket_types_excluded = ["run out", "retired hurt", "obstructing the field"]
    wickets_df = df[
        (df["is_wicket"] == 1)
        & (~df["dismissal_kind"].isin(wicket_types_excluded))
    ]
    wickets = (
        wickets_df.groupby("bowler")
        .size()
        .reset_index(name="wickets")
    )

    # Merge
    stats = runs.merge(balls, on="bowler", how="left")
    stats = stats.merge(wickets, on="bowler", how="left")
    stats["wickets"] = stats["wickets"].fillna(0).astype(int)
    stats["balls_bowled"] = stats["balls_bowled"].fillna(0).astype(int)

    # Overs
    stats["overs"] = (stats["balls_bowled"] / 6).round(2)

    # Economy Rate = Runs Conceded / Overs
    stats["economy"] = np.where(
        stats["overs"] > 0,
        (stats["runs_conceded"] / stats["overs"]).round(2),
        0.0,
    )

    return stats.sort_values("wickets", ascending=False).reset_index(drop=True)


def top_bowlers(
    deliveries: pd.DataFrame,
    metric: str = "wickets",
    n: int = 10,
    min_balls: int = 300,
) -> pd.DataFrame:
    """Return top-N bowlers ranked by a given metric.

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball data.
    metric : str
        One of 'wickets', 'economy' (for economy, lower is better → ascending sort).
    n : int
        Number of bowlers to return.
    min_balls : int
        Minimum balls bowled to qualify.

    Returns
    -------
    pd.DataFrame
        Top-N bowlers.
    """
    stats = bowler_stats(deliveries)
    qualified = stats[stats["balls_bowled"] >= min_balls]
    if metric == "economy":
        return qualified.nsmallest(n, metric).reset_index(drop=True)
    return qualified.nlargest(n, metric).reset_index(drop=True)


def bowler_season_wickets(
    merged: pd.DataFrame,
    player: str,
) -> pd.DataFrame:
    """Get season-wise wicket tally for a specific bowler.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged data with 'season' column.
    player : str
        Bowler name.

    Returns
    -------
    pd.DataFrame
        Columns: [season, wickets].
    """
    wicket_types_excluded = ["run out", "retired hurt", "obstructing the field"]
    player_data = merged[
        (merged["bowler"] == player)
        & (merged["is_wicket"] == 1)
        & (~merged["dismissal_kind"].isin(wicket_types_excluded))
    ]
    season_wkts = (
        player_data.groupby("season")
        .size()
        .reset_index(name="wickets")
    )
    return season_wkts


# ===================================================================
# Venue & Team Insights
# ===================================================================

def venue_avg_first_innings_score(merged: pd.DataFrame) -> pd.DataFrame:
    """Average first-innings total per venue.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged match + delivery data.

    Returns
    -------
    pd.DataFrame
        Columns: [venue, avg_score, matches].
    """
    first_innings = merged[merged["inning"] == 1]
    match_totals = (
        first_innings.groupby(["match_id", "venue"])["total_runs"]
        .sum()
        .reset_index()
    )
    venue_stats = (
        match_totals.groupby("venue")
        .agg(avg_score=("total_runs", "mean"), matches=("total_runs", "count"))
        .reset_index()
    )
    venue_stats["avg_score"] = venue_stats["avg_score"].round(1)
    return venue_stats.sort_values("avg_score", ascending=False).reset_index(
        drop=True
    )


def toss_match_win_correlation(matches: pd.DataFrame) -> pd.DataFrame:
    """Toss-win vs. match-win correlation analysis.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: [toss_decision, toss_wins, match_wins, win_pct].
    """
    df = matches[matches["winner"] != "No Result"].copy()
    df["toss_match_win"] = (df["toss_winner"] == df["winner"]).astype(int)

    overall = df["toss_match_win"].mean() * 100

    by_decision = (
        df.groupby("toss_decision")
        .agg(
            toss_wins=("toss_match_win", "count"),
            match_wins=("toss_match_win", "sum"),
        )
        .reset_index()
    )
    by_decision["win_pct"] = (
        (by_decision["match_wins"] / by_decision["toss_wins"]) * 100
    ).round(1)

    # Add overall row
    overall_row = pd.DataFrame(
        [
            {
                "toss_decision": "Overall",
                "toss_wins": len(df),
                "match_wins": df["toss_match_win"].sum(),
                "win_pct": round(overall, 1),
            }
        ]
    )
    return pd.concat([by_decision, overall_row], ignore_index=True)


def team_win_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Win count and win percentage for each team.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: [team, matches_played, wins, win_pct].
    """
    valid = matches[matches["winner"] != "No Result"]

    # Matches played (team appeared as team1 or team2)
    team1_counts = valid["team1"].value_counts()
    team2_counts = valid["team2"].value_counts()
    played = (team1_counts.add(team2_counts, fill_value=0)).astype(int)

    # Wins
    wins = valid["winner"].value_counts()

    stats = pd.DataFrame({"matches_played": played, "wins": wins}).fillna(0)
    stats["wins"] = stats["wins"].astype(int)
    stats["win_pct"] = ((stats["wins"] / stats["matches_played"]) * 100).round(1)
    stats = stats.reset_index().rename(columns={"index": "team"})

    return stats.sort_values("wins", ascending=False).reset_index(drop=True)


def season_performance(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Season-wise win record for a specific team.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches DataFrame.
    team : str
        Team name (standardized).

    Returns
    -------
    pd.DataFrame
        Columns: [season, played, wins, win_pct].
    """
    team_matches = matches[
        (matches["team1"] == team) | (matches["team2"] == team)
    ].copy()

    played = team_matches.groupby("season").size().reset_index(name="played")
    wins = (
        team_matches[team_matches["winner"] == team]
        .groupby("season")
        .size()
        .reset_index(name="wins")
    )

    result = played.merge(wins, on="season", how="left")
    result["wins"] = result["wins"].fillna(0).astype(int)
    result["win_pct"] = ((result["wins"] / result["played"]) * 100).round(1)

    return result


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """Head-to-head record between two teams.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches DataFrame.
    team_a, team_b : str
        Team names.

    Returns
    -------
    dict
        Keys: team_a_wins, team_b_wins, no_result, total.
    """
    h2h = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ]

    return {
        "total": len(h2h),
        f"{team_a}_wins": int((h2h["winner"] == team_a).sum()),
        f"{team_b}_wins": int((h2h["winner"] == team_b).sum()),
        "no_result": int((h2h["winner"] == "No Result").sum()),
    }


def get_all_teams(matches: pd.DataFrame) -> list[str]:
    """Return sorted list of all unique team names.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches DataFrame.

    Returns
    -------
    list[str]
        Alphabetically sorted team names.
    """
    teams = set(matches["team1"].unique()) | set(matches["team2"].unique())
    return sorted(teams)


def get_all_batsmen(deliveries: pd.DataFrame, min_balls: int = 100) -> list[str]:
    """Return sorted list of batsmen who have faced at least *min_balls*.

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball data.
    min_balls : int
        Minimum balls faced to be included.

    Returns
    -------
    list[str]
        Sorted batsman names.
    """
    counts = deliveries.groupby("batter").size()
    return sorted(counts[counts >= min_balls].index.tolist())


def get_all_bowlers(deliveries: pd.DataFrame, min_balls: int = 100) -> list[str]:
    """Return sorted list of bowlers who have bowled at least *min_balls*.

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball data.
    min_balls : int
        Minimum balls bowled to be included.

    Returns
    -------
    list[str]
        Sorted bowler names.
    """
    counts = deliveries.groupby("bowler").size()
    return sorted(counts[counts >= min_balls].index.tolist())


def matches_per_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Count of matches played each season.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: [season, matches].
    """
    return (
        matches.groupby("season")
        .size()
        .reset_index(name="matches")
        .sort_values("season")
    )


def top_run_scorers(deliveries: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N all-time run scorers.

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball data.
    n : int
        Number of players.

    Returns
    -------
    pd.DataFrame
        Columns: [batter, runs].
    """
    return (
        deliveries.groupby("batter")["batsman_runs"]
        .sum()
        .nlargest(n)
        .reset_index()
        .rename(columns={"batsman_runs": "runs"})
    )


def total_sixes_fours(deliveries: pd.DataFrame) -> dict:
    """Total number of 4s and 6s across all IPL matches.

    Parameters
    ----------
    deliveries : pd.DataFrame
        Ball-by-ball data.

    Returns
    -------
    dict
        Keys: fours, sixes.
    """
    return {
        "fours": int((deliveries["batsman_runs"] == 4).sum()),
        "sixes": int((deliveries["batsman_runs"] == 6).sum()),
    }
