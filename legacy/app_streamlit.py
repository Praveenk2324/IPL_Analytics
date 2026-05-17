"""
IPL Sports Analytics & Predictor — Streamlit Dashboard.

A comprehensive dark-mode dashboard with four sections:
Home, Team Analysis, Player Stats, and Score Predictor.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import get_clean_data
from src.analysis import (
    batsman_stats, top_batsmen, batsman_season_runs,
    bowler_stats, top_bowlers, bowler_season_wickets,
    venue_avg_first_innings_score, toss_match_win_correlation,
    team_win_stats, season_performance, head_to_head,
    get_all_teams, get_all_batsmen, get_all_bowlers,
    matches_per_season, top_run_scorers, total_sixes_fours,
)
from src.models import (
    prepare_ml_features, train_model, predict_score,
    save_model, load_model,
)

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Sports Analytics & Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Dark-Mode Friendly) ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-header {
    text-align: center;
    color: #a0aec0;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.25);
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-label {
    font-size: 0.9rem;
    color: #a0aec0;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(102,126,234,0.3);
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(26,26,46,0.95) 0%, rgba(22,22,40,0.98) 100%);
}

.stSelectbox > div > div {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly Theme ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=50, b=40),
)

COLORS = px.colors.sequential.Plasma_r


# ── Data Loading (cached) ───────────────────────────────────────────
@st.cache_data(show_spinner="Loading IPL data...")
def load_all_data():
    """Load and cache cleaned data."""
    return get_clean_data()


@st.cache_resource(show_spinner="Training prediction model...")
def get_trained_model(_merged):
    """Train and cache the score prediction model."""
    model_path = "models/score_predictor.joblib"
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception:
            pass
    features = prepare_ml_features(_merged)
    result = train_model(features)
    save_model(result, model_path)
    return result


# ── Sidebar Navigation ──────────────────────────────────────────────
st.sidebar.markdown("## 🏏 IPL Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Team Analysis", "🏅 Player Stats", "🎯 Score Predictor"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align:center;color:#718096;font-size:0.8rem;'>"
    "Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)

# ── Load Data ────────────────────────────────────────────────────────
try:
    matches, deliveries, merged = load_all_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

teams = get_all_teams(matches)

# =====================================================================
# PAGE: HOME
# =====================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">IPL Sports Analytics & Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analytics across all IPL seasons — powered by data science & ML</p>', unsafe_allow_html=True)

    # KPI Cards
    boundary_stats = total_sixes_fours(deliveries)
    seasons = matches["season"].nunique()
    total_matches = len(matches)

    cols = st.columns(4)
    kpis = [
        ("Total Matches", f"{total_matches:,}"),
        ("Seasons", str(seasons)),
        ("Total 6s", f"{boundary_stats['sixes']:,}"),
        ("Total 4s", f"{boundary_stats['fours']:,}"),
    ]
    for col, (label, value) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # Charts row 1
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">📅 Matches Per Season</div>', unsafe_allow_html=True)
        mps = matches_per_season(matches)
        fig = px.bar(
            mps, x="season", y="matches",
            color="matches", color_continuous_scale="Plasma",
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0, hovertemplate="Season: %{x}<br>Matches: %{y}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">🏆 Top 10 Run Scorers (All Time)</div>', unsafe_allow_html=True)
        top_runs = top_run_scorers(deliveries, n=10)
        fig = px.bar(
            top_runs, x="runs", y="batter", orientation="h",
            color="runs", color_continuous_scale="Viridis",
        )
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig.update_traces(hovertemplate="%{y}: %{x} runs")
        st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">🥧 Toss Decision Distribution</div>', unsafe_allow_html=True)
        toss_dist = matches["toss_decision"].value_counts().reset_index()
        toss_dist.columns = ["decision", "count"]
        fig = px.pie(toss_dist, names="decision", values="count", color_discrete_sequence=["#667eea", "#764ba2"], hole=0.45)
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value}")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">🎯 Toss Win → Match Win %</div>', unsafe_allow_html=True)
        toss_corr = toss_match_win_correlation(matches)
        fig = px.bar(
            toss_corr, x="toss_decision", y="win_pct",
            color="toss_decision", color_discrete_sequence=["#667eea", "#764ba2", "#e53e3e"],
            text="win_pct",
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, yaxis_title="Win %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE: TEAM ANALYSIS
# =====================================================================
elif page == "📊 Team Analysis":
    st.markdown('<h1 class="main-header">Team Analysis</h1>', unsafe_allow_html=True)

    selected_team = st.selectbox("Select a Team", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)

    # Win Stats
    win_data = team_win_stats(matches)
    team_row = win_data[win_data["team"] == selected_team]

    if not team_row.empty:
        cols = st.columns(3)
        tr = team_row.iloc[0]
        for col, (label, val) in zip(cols, [
            ("Matches Played", f"{tr['matches_played']}"),
            ("Wins", f"{tr['wins']}"),
            ("Win %", f"{tr['win_pct']}%"),
        ]):
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                f'<div class="kpi-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">📈 Season-wise Performance</div>', unsafe_allow_html=True)
        sp = season_performance(matches, selected_team)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sp["season"], y=sp["wins"], mode="lines+markers",
            name="Wins", line=dict(color="#667eea", width=3),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Bar(
            x=sp["season"], y=sp["played"], name="Played",
            marker_color="rgba(118,75,162,0.3)",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">🥧 Win / Loss Split</div>', unsafe_allow_html=True)
        if not team_row.empty:
            tr = team_row.iloc[0]
            losses = tr["matches_played"] - tr["wins"]
            pie_data = pd.DataFrame({"result": ["Wins", "Losses"], "count": [tr["wins"], losses]})
            fig = px.pie(pie_data, names="result", values="count", color_discrete_sequence=["#48bb78", "#fc8181"], hole=0.45)
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(textinfo="percent+value")
            st.plotly_chart(fig, use_container_width=True)

    # Head to Head
    st.markdown('<div class="section-title">⚔️ Head-to-Head</div>', unsafe_allow_html=True)
    opponent = st.selectbox("Select Opponent", [t for t in teams if t != selected_team])
    h2h = head_to_head(matches, selected_team, opponent)
    cols = st.columns(3)
    cols[0].metric(f"{selected_team}", h2h.get(f"{selected_team}_wins", 0))
    cols[1].metric("Total Matches", h2h["total"])
    cols[2].metric(f"{opponent}", h2h.get(f"{opponent}_wins", 0))

    # Venue insights
    st.markdown('<div class="section-title">🏟️ Top Venues by Avg 1st Innings Score</div>', unsafe_allow_html=True)
    venue_scores = venue_avg_first_innings_score(merged)
    top_venues = venue_scores[venue_scores["matches"] >= 10].head(15)
    fig = px.bar(
        top_venues, x="avg_score", y="venue", orientation="h",
        color="avg_score", color_continuous_scale="Inferno",
        text="avg_score",
    )
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=500)
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE: PLAYER STATS
# =====================================================================
elif page == "🏅 Player Stats":
    st.markdown('<h1 class="main-header">Player Stats</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏏 Batsman", "🎳 Bowler"])

    # ── Batsman Tab ──────────────────────────────────────────────────
    with tab1:
        all_batsmen = get_all_batsmen(deliveries, min_balls=100)
        selected_bat = st.selectbox("Select Batsman", all_batsmen, index=all_batsmen.index("V Kohli") if "V Kohli" in all_batsmen else 0)

        bat_stats = batsman_stats(deliveries)
        player_row = bat_stats[bat_stats["batter"] == selected_bat]

        if not player_row.empty:
            pr = player_row.iloc[0]
            cols = st.columns(4)
            for col, (label, val) in zip(cols, [
                ("Runs", f"{int(pr['runs']):,}"),
                ("Balls Faced", f"{int(pr['balls_faced']):,}"),
                ("Strike Rate", f"{pr['strike_rate']}"),
                ("Average", f"{pr['average']}"),
            ]):
                col.markdown(
                    f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                    f'<div class="kpi-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">📈 Runs Per Season</div>', unsafe_allow_html=True)
            sr = batsman_season_runs(merged, selected_bat)
            if not sr.empty:
                fig = px.bar(sr, x="season", y="runs", color="runs", color_continuous_scale="Plasma", text="runs")
                fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No season data available.")

        with c2:
            st.markdown('<div class="section-title">🏆 Top 10 by Strike Rate</div>', unsafe_allow_html=True)
            top_sr = top_batsmen(deliveries, metric="strike_rate", n=10, min_balls=500)
            fig = px.bar(top_sr, x="strike_rate", y="batter", orientation="h", color="strike_rate", color_continuous_scale="Viridis")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Bowler Tab ───────────────────────────────────────────────────
    with tab2:
        all_bowlers = get_all_bowlers(deliveries, min_balls=100)
        selected_bowl = st.selectbox("Select Bowler", all_bowlers, index=all_bowlers.index("JJ Bumrah") if "JJ Bumrah" in all_bowlers else 0)

        bowl_stats = bowler_stats(deliveries)
        bowler_row = bowl_stats[bowl_stats["bowler"] == selected_bowl]

        if not bowler_row.empty:
            br = bowler_row.iloc[0]
            cols = st.columns(4)
            for col, (label, val) in zip(cols, [
                ("Wickets", f"{int(br['wickets'])}"),
                ("Economy", f"{br['economy']}"),
                ("Overs", f"{br['overs']}"),
                ("Runs Conceded", f"{int(br['runs_conceded']):,}"),
            ]):
                col.markdown(
                    f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                    f'<div class="kpi-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">📈 Wickets Per Season</div>', unsafe_allow_html=True)
            sw = bowler_season_wickets(merged, selected_bowl)
            if not sw.empty:
                fig = px.bar(sw, x="season", y="wickets", color="wickets", color_continuous_scale="Magma", text="wickets")
                fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No season data available.")

        with c2:
            st.markdown('<div class="section-title">🏆 Top 10 by Wickets</div>', unsafe_allow_html=True)
            top_wk = top_bowlers(deliveries, metric="wickets", n=10, min_balls=300)
            fig = px.bar(top_wk, x="wickets", y="bowler", orientation="h", color="wickets", color_continuous_scale="Cividis")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE: SCORE PREDICTOR
# =====================================================================
elif page == "🎯 Score Predictor":
    st.markdown('<h1 class="main-header">First Innings Score Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict the final first-innings score using a Random Forest model</p>', unsafe_allow_html=True)

    # Train / load model
    try:
        model_dict = get_trained_model(merged)
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        st.stop()

    model = model_dict["model"]
    encoders = model_dict["encoders"]
    metrics = model_dict["metrics"]

    # Model metrics display
    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
    mc = st.columns(3)
    for col, (label, val) in zip(mc, [
        ("MAE", metrics["MAE"]),
        ("RMSE", metrics["RMSE"]),
        ("R² Score", metrics["R2"]),
    ]):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-title">🎮 Predict Score</div>', unsafe_allow_html=True)

    # Input controls
    c1, c2 = st.columns(2)
    with c1:
        batting_team = st.selectbox("Batting Team", teams, key="bat_pred")
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], key="bowl_pred")
        venues = sorted(merged["venue"].dropna().unique().tolist())
        venue = st.selectbox("Venue", venues, key="venue_pred")

    with c2:
        overs = st.slider("Overs Completed", 5, 20, 10, key="overs_pred")
        current_score = st.slider("Current Score", 0, 350, 80, key="score_pred")
        wickets = st.slider("Wickets Fallen", 0, 9, 2, key="wkt_pred")

    if st.button("🔮 Predict Final Score", use_container_width=True):
        predicted = predict_score(
            model, encoders,
            current_score=current_score,
            wickets_fallen=wickets,
            overs_completed=overs,
            venue=venue,
            batting_team=batting_team,
            bowling_team=bowling_team,
        )

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted,
            delta={"reference": current_score, "increasing": {"color": "#48bb78"}},
            title={"text": "Predicted Final Score", "font": {"size": 20, "color": "#e2e8f0"}},
            number={"font": {"size": 60, "color": "#667eea"}},
            gauge={
                "axis": {"range": [0, 300], "tickcolor": "#e2e8f0"},
                "bar": {"color": "#667eea"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 120], "color": "rgba(229,62,62,0.2)"},
                    {"range": [120, 180], "color": "rgba(236,201,75,0.2)"},
                    {"range": [180, 300], "color": "rgba(72,187,120,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#fc8181", "width": 3},
                    "thickness": 0.8,
                    "value": current_score,
                },
            },
        ))
        gauge_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}
        fig.update_layout(
            **gauge_layout, height=350,
            margin=dict(l=30, r=30, t=60, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        remaining_runs = predicted - current_score
        remaining_overs = 20 - overs
        req_rate = remaining_runs / remaining_overs if remaining_overs > 0 else 0
        st.markdown(
            f"""
            <div class="kpi-card" style="margin-top:1rem;">
                <p style="font-size:1.1rem;color:#e2e8f0;">
                    🏏 <b>{batting_team}</b> vs <b>{bowling_team}</b> at <b>{venue}</b><br>
                    After <b>{overs}</b> overs: <b>{current_score}/{wickets}</b><br>
                    📈 Predicted Final Score: <b style="color:#667eea;">{predicted}</b><br>
                    🎯 Runs needed in remaining {remaining_overs} overs: <b>{remaining_runs}</b>
                    (Required Rate: <b>{req_rate:.1f}</b>)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
