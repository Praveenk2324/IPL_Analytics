"""
IPL Analytics — Data Cleaning & Preprocessing Utilities.

This module provides functions for loading, cleaning, standardizing,
and merging IPL match and delivery datasets for downstream analysis
and machine learning tasks.
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

# Venue → City fallback mapping (for the 51 null cities)
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


def load_data(
    matches_path: str = "data/matches.csv",
    deliveries_path: str = "data/deliveries.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load match and delivery CSVs with robust error handling.

    Parameters
    ----------
    matches_path : str
        File path to the matches CSV.
    deliveries_path : str
        File path to the deliveries CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (matches_df, deliveries_df).

    Raises
    ------
    FileNotFoundError
        If either CSV file is missing.
    pd.errors.ParserError
        If either CSV is malformed.
    """
    try:
        matches = pd.read_csv(matches_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Matches CSV not found at '{matches_path}'. "
            "Please place it in the data/ directory."
        )
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError(
            f"Failed to parse matches CSV: {exc}"
        ) from exc

    try:
        deliveries = pd.read_csv(deliveries_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Deliveries CSV not found at '{deliveries_path}'. "
            "Please place it in the data/ directory."
        )
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError(
            f"Failed to parse deliveries CSV: {exc}"
        ) from exc

    return matches, deliveries


def standardize_team_names(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Replace legacy team names with current franchise names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing team-name columns.
    columns : list[str] | None
        Column names to standardize. If *None*, auto-detects common
        team-name columns present in the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized team names.
    """
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
    """Fill missing values in the matches DataFrame.

    Strategy:
      - ``city``: infer from venue using *VENUE_CITY_MAP*; remaining → 'Unknown'.
      - ``winner``: fill with 'No Result' (tied / no-result matches).
      - ``player_of_match``: fill with 'Not Awarded'.
      - ``result_margin``: fill with 0.
      - ``method``: fill with 'Normal' (non-DLS matches).

    Parameters
    ----------
    matches : pd.DataFrame
        Raw matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned matches DataFrame.
    """
    matches = matches.copy()

    # Infer city from venue
    if "city" in matches.columns and "venue" in matches.columns:
        mask = matches["city"].isnull()
        matches.loc[mask, "city"] = matches.loc[mask, "venue"].map(VENUE_CITY_MAP)
        matches["city"] = matches["city"].fillna("Unknown")

    matches["winner"] = matches["winner"].fillna("No Result")
    matches["player_of_match"] = matches["player_of_match"].fillna("Not Awarded")
    matches["result_margin"] = matches["result_margin"].fillna(0)
    matches["method"] = matches["method"].fillna("Normal")

    return matches


def merge_data(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ball-by-ball deliveries with match metadata.

    Adds ``season``, ``city``, ``venue``, ``date``, and ``match_type``
    columns from the matches table to every delivery record.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches DataFrame.
    deliveries : pd.DataFrame
        Deliveries DataFrame (team names already standardized).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    merge_cols = ["id", "season", "city", "venue", "date", "match_type"]
    merged = deliveries.merge(
        matches[merge_cols],
        left_on="match_id",
        right_on="id",
        how="left",
    )
    merged.drop(columns=["id"], inplace=True)
    return merged


def get_clean_data(
    matches_path: str = "data/matches.csv",
    deliveries_path: str = "data/deliveries.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-call orchestrator: load → clean → merge.

    Parameters
    ----------
    matches_path : str
        Path to matches CSV.
    deliveries_path : str
        Path to deliveries CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (matches_clean, deliveries_clean, merged).
    """
    matches, deliveries = load_data(matches_path, deliveries_path)

    matches = standardize_team_names(matches)
    matches = handle_missing_values(matches)

    deliveries = standardize_team_names(deliveries)

    merged = merge_data(matches, deliveries)

    return matches, deliveries, merged
