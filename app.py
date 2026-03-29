import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from analysis import preprocess, season_winners_trend
from ml_models import (
    build_prediction_row,
    explain_prediction_from_global_importance,
    predict_match_winner,
    train_and_evaluate_models,
)


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IPL Analytics Dashboard", layout="wide", page_icon="🏏")


# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7FB;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

/* Remove top padding */
.block-container {
    padding-top: 1rem;
}

/* Title */
h1, h2, h3 {
    color: #111827;
}

/* Metric Cards */
.metric-card {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    text-align: center;
}

/* Buttons */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #1D4ED8;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #6B7280;
    font-size: 13px;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "data", "matches.csv")
    df = pd.read_csv(path)
    df = preprocess(df)
    return df

df = load_data()


# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>🏏 IPL Analytics Dashboard</h1>
<p style='text-align:center; color:#6B7280; font-size:18px;'>
Team Performance & Winning Patterns (2008–2019)
</p>
""", unsafe_allow_html=True)


# ---------------- TOP METRICS ----------------
total_matches = len(df)
teams = pd.concat([df["team1"], df["team2"]]).unique()
seasons = df["match_year"].nunique()
venues = df["venue"].nunique()

c1, c2, c3, c4 = st.columns(4)

def metric(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:14px;color:#6B7280;">{label}</div>
        <div style="font-size:28px;font-weight:700;color:#111827;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

with c1: metric("Total Matches", total_matches)
with c2: metric("Total Teams", len(teams))
with c3: metric("Seasons Covered", seasons)
with c4: metric("Venues", venues)

st.divider()

# ---------------- ML SETTINGS (shared) ----------------
# Shared split mode so all tabs use the same trained ML model.
ml_split_mode = st.selectbox(
    "ML Split Mode",
    options=["latest_season", "last_two_seasons"],
    index=0,
    help="latest_season: train on earlier seasons, test on latest season. last_two_seasons: test on last two seasons.",
)

@st.cache_resource
def train_ml_cached(split_mode_input: str):
    # Cached training so every tab can use ML predictions without retraining repeatedly.
    return train_and_evaluate_models(df, split_mode=split_mode_input)


def ml_predict_winner(team1: str, team2: str, venue: str, toss_winner: str, toss_decision: str):
    """
    Run the ML winner predictor for a single match scenario.
    Model target is binary: probability that `team1` wins.
    """
    ml = train_ml_cached(ml_split_mode)
    raw_row = build_prediction_row(
        raw_df=df,
        team1=team1,
        team2=team2,
        venue=venue,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
    )
    pred, proba = predict_match_winner(
        ml["best_model"],
        raw_row,
        team1_name=team1,
        team2_name=team2,
    )
    try:
        explanation_lines = explain_prediction_from_global_importance(ml["feature_importance_df"], raw_row)
    except Exception:
        explanation_lines = []
    return pred, proba, explanation_lines, ml["best_model_name"]

#
# ---------------- TABS NAVIGATION ----------------

# ---------------- TABS NAVIGATION ----------------
tab_home, tab_team, tab_toss, tab_venue, tab_players, tab_season, tab_ml = st.tabs(

    [
        "Home",
        "Team Performance",
        "Toss Analysis",
        "Venue Analysis",
        "Player Awards",
        "Season Trends",
        "ML Prediction Lab",
    ]
)


# ---------------- HOME ----------------
with tab_home:
    st.subheader("Overview")

    st.write("""
This dashboard analyzes historical IPL match data and identifies:
- Team winning patterns
- Toss impact on results
- Venue advantages
- Player performances
    """)

    matches_by_team = pd.concat([df["team1"], df["team2"]]).value_counts().reset_index()
    matches_by_team.columns = ["team", "matches"]

    top_n = st.slider("Top N teams by matches", min_value=3, max_value=30, value=10, step=1)
    fig = px.bar(
        matches_by_team.head(int(top_n)),
        x="team",
        y="matches",
        color="matches",
        color_continuous_scale="Blues",
        title="Top Teams by Matches Played",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ML widget for this tab (interactive prediction)
    st.divider()
    st.subheader("ML Quick Prediction")
    all_teams = sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique())
    all_venues = sorted(df["venue"].dropna().unique()) if "venue" in df.columns else []
    toss_choices = sorted(df["toss_decision"].dropna().unique()) if "toss_decision" in df.columns else ["bat", "field"]

    team_a = st.selectbox("Team A (Team1 in ML)", all_teams, index=0, key="home_team_a")
    team_b_default = 1 if len(all_teams) > 1 else 0
    team_b = st.selectbox("Team B (Team2 in ML)", all_teams, index=team_b_default, key="home_team_b")
    venue_a = st.selectbox("Venue", all_venues, index=0 if all_venues else None, key="home_venue")
    toss_dec = st.selectbox("Toss Decision", toss_choices, index=0, key="home_toss_dec")
    toss_win = st.selectbox("Toss Winner", [team_a, team_b], index=0, key="home_toss_win")

    if st.button("Predict Winner (ML)", key="home_predict"):
        if team_a == team_b:
            st.warning("Team A and Team B must be different.")
        elif not all_venues:
            st.warning("Venue information is missing from your dataset.")
        else:
            pred, proba, explanation_lines, best_model_name = ml_predict_winner(
                team1=team_a,
                team2=team_b,
                venue=venue_a,
                toss_winner=toss_win,
                toss_decision=toss_dec,
            )
            st.success(f"Predicted Winner: {pred} (P(TeamA win)={proba:.2%}) | Model: {best_model_name}")
            if explanation_lines:
                with st.expander("ML Explanation (top factors / engineered values)"):
                    for line in explanation_lines[:5]:
                        st.write(f"- {line}")


# ---------------- TEAM PERFORMANCE ----------------
with tab_team:
    st.subheader("Team Performance")

    teams_list = sorted(teams)
    team = st.selectbox("Select Team", teams_list)

    matches_played = df[(df["team1"] == team) | (df["team2"] == team)].shape[0]
    wins = df[df["winner"] == team].shape[0]
    win_pct = (wins / matches_played) * 100 if matches_played else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches Played", matches_played)
    c2.metric("Wins", wins)
    c3.metric("Win %", f"{win_pct:.1f}%")

    wins_by_season = df[df["winner"] == team].groupby("match_year").size().reset_index(name="wins")

    fig = px.bar(wins_by_season, x="match_year", y="wins",
                 title=f"{team} Wins Per Season",
                 color_discrete_sequence=["#10B981"])
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("ML Prediction: Selected Team vs Opponent")
    all_teams = sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique())
    all_venues = sorted(df["venue"].dropna().unique()) if "venue" in df.columns else []
    toss_choices = sorted(df["toss_decision"].dropna().unique()) if "toss_decision" in df.columns else ["bat", "field"]

    opponent_default_idx = 0 if (len(all_teams) < 2) else 1
    opponent = st.selectbox(
        "Opponent (Team B)",
        [t for t in all_teams if t != team] if len(all_teams) else all_teams,
        index=0 if len(all_teams) > 1 else 0,
    )
    venue_b = st.selectbox("Venue", all_venues, index=0 if all_venues else None, key="team_venue_ml")
    toss_dec = st.selectbox("Toss Decision", toss_choices, index=0, key="team_toss_dec_ml")
    toss_win_b = st.selectbox("Toss Winner", [team, opponent], index=0, key="team_toss_win_ml")

    if st.button("Predict Winner (ML)", key="team_predict_ml"):
        if not all_venues:
            st.warning("Venue information is missing from your dataset.")
        elif team == opponent:
            st.warning("Selected team and opponent must be different.")
        else:
            pred, proba, explanation_lines, best_model_name = ml_predict_winner(
                team1=team,
                team2=opponent,
                venue=venue_b,
                toss_winner=toss_win_b,
                toss_decision=toss_dec,
            )
            st.success(f"Predicted Winner: {pred} (P(Team1 win)={proba:.2%}) | Model: {best_model_name}")
            if explanation_lines:
                with st.expander("ML Explanation (top factors / engineered values)"):
                    for line in explanation_lines[:5]:
                        st.write(f"- {line}")


# ---------------- TOSS ANALYSIS ----------------
with tab_toss:
    st.subheader("Toss Impact")

    if "match_year" in df.columns:
        years = pd.to_numeric(df["match_year"], errors="coerce")
        min_y, max_y = int(years.min()), int(years.max())
    else:
        min_y, max_y = 0, 0

    year_range = st.slider(
        "Year range",
        min_value=min_y,
        max_value=max_y,
        value=(min_y, max_y),
        step=1,
        key="toss_year_range",
    )
    df_t = df.copy()
    if min_y != max_y:
        df_t = df_t[(pd.to_numeric(df_t["match_year"], errors="coerce") >= year_range[0]) & (pd.to_numeric(df_t["match_year"], errors="coerce") <= year_range[1])]

    toss_decision_choice = "All"
    if "toss_decision" in df_t.columns:
        toss_decision_choice = st.selectbox("Toss decision filter", ["All"] + sorted(df_t["toss_decision"].dropna().unique()))
        if toss_decision_choice != "All":
            df_t = df_t[df_t["toss_decision"] == toss_decision_choice]

    if "toss_winner" in df_t.columns and "winner" in df_t.columns:
        df_t["toss_win_match"] = np.where(df_t["toss_winner"] == df_t["winner"], "Won Match", "Lost Match")
        toss_data = df_t["toss_win_match"].value_counts().reset_index()
        toss_data.columns = ["Result", "Count"]

        fig = px.pie(
            toss_data,
            names="Result",
            values="Count",
            title="Does Toss Winner Win the Match?",
            color_discrete_sequence=["#10B981", "#EF4444"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("toss_winner/winner columns not found in dataset.")

    st.divider()
    st.subheader("ML Prediction: Probability Toss Winner Wins")
    all_teams = sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique())
    all_venues = sorted(df["venue"].dropna().unique()) if "venue" in df.columns else []
    toss_choices = sorted(df["toss_decision"].dropna().unique()) if "toss_decision" in df.columns else ["bat", "field"]

    toss_team = st.selectbox("Toss Winner Team (Team1 in ML)", all_teams, index=0, key="toss_ml_toss_team")
    opponent = st.selectbox(
        "Opponent Team (Team2 in ML)",
        [t for t in all_teams if t != toss_team] if len(all_teams) > 1 else all_teams,
        index=0,
        key="toss_ml_opponent",
    )
    venue_x = st.selectbox("Venue", all_venues, index=0 if all_venues else None, key="toss_ml_venue")
    toss_dec = st.selectbox("Toss Decision", toss_choices, index=0, key="toss_ml_toss_decision")

    if st.button("Predict (ML)", key="toss_ml_predict"):
        if not all_venues:
            st.warning("Venue information is missing from your dataset.")
        else:
            pred, proba, explanation_lines, best_model_name = ml_predict_winner(
                team1=toss_team,
                team2=opponent,
                venue=venue_x,
                toss_winner=toss_team,
                toss_decision=toss_dec,
            )
            st.success(f"Predicted Winner: {pred} | P(TossWinner wins)={proba:.2%} | {best_model_name}")
            if explanation_lines:
                with st.expander("ML Explanation"):
                    for line in explanation_lines[:5]:
                        st.write(f"- {line}")


# ---------------- VENUE ANALYSIS ----------------
with tab_venue:
    st.subheader("Venue Advantage")

    venue = st.selectbox("Select Venue", sorted(df["venue"].unique()))
    venue_df = df[df["venue"] == venue]

    wins = venue_df["winner"].value_counts().reset_index()
    wins.columns = ["Team", "Wins"]

    top_n = st.slider("Top N winning teams to show", min_value=3, max_value=25, value=10, step=1)
    fig = px.bar(
        wins.head(int(top_n)),
        x="Wins",
        y="Team",
        orientation="h",
        title=f"Wins at {venue}",
        color_discrete_sequence=["#F59E0B"],
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("ML Prediction: Venue Match")
    teams_list = sorted(teams)
    all_venues = sorted(df["venue"].dropna().unique()) if "venue" in df.columns else []
    toss_choices = sorted(df["toss_decision"].dropna().unique()) if "toss_decision" in df.columns else ["bat", "field"]

    team_a = st.selectbox("Team A (Team1 in ML)", teams_list, index=0, key="venue_team_a")
    team_b = st.selectbox(
        "Team B (Team2 in ML)",
        [t for t in teams_list if t != team_a] if len(teams_list) > 1 else teams_list,
        index=0 if len(teams_list) > 1 else 0,
        key="venue_team_b",
    )
    toss_win = st.selectbox("Toss Winner", [team_a, team_b], index=0, key="venue_toss_win")
    toss_dec = st.selectbox("Toss Decision", toss_choices, index=0, key="venue_toss_dec")

    if st.button("Predict (ML)", key="venue_predict"):
        if not all_venues:
            st.warning("Venue information is missing from your dataset.")
        elif team_a == team_b:
            st.warning("Team A and Team B must be different.")
        else:
            pred, proba, explanation_lines, best_model_name = ml_predict_winner(
                team1=team_a,
                team2=team_b,
                venue=venue,
                toss_winner=toss_win,
                toss_decision=toss_dec,
            )
            st.success(f"Predicted Winner: {pred} | P(Team1 win)={proba:.2%} | {best_model_name}")
            if explanation_lines:
                with st.expander("ML Explanation"):
                    for line in explanation_lines[:5]:
                        st.write(f"- {line}")


# ---------------- PLAYER AWARDS ----------------
with tab_players:
    st.subheader("Top Players")
    if "player_of_match" not in df.columns:
        st.warning("player_of_match column not found in dataset.")
    else:
        if "match_year" in df.columns:
            years = pd.to_numeric(df["match_year"], errors="coerce")
            min_y, max_y = int(years.min()), int(years.max())
            year_range = st.slider(
                "Year range",
                min_value=min_y,
                max_value=max_y,
                value=(min_y, max_y),
                step=1,
                key="players_year_range",
            )
            df_p = df[
                (pd.to_numeric(df["match_year"], errors="coerce") >= year_range[0])
                & (pd.to_numeric(df["match_year"], errors="coerce") <= year_range[1])
            ].copy()
        else:
            df_p = df.copy()

        top_n_players = st.slider("Top N players (bar)", min_value=3, max_value=30, value=10, step=1, key="players_topn")
        table_n = st.slider("Rows in table", min_value=5, max_value=60, value=20, step=5, key="players_tablen")

        pom = df_p["player_of_match"].value_counts().reset_index()
        pom.columns = ["Player", "Awards"]

        fig = px.bar(
            pom.head(int(top_n_players)),
            x="Awards",
            y="Player",
            orientation="h",
            title=f"Top {top_n_players} Player of the Match Winners",
            color_discrete_sequence=["#F59E0B"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pom.head(int(table_n)))

        st.divider()
        st.subheader("ML Prediction: Expected Winner + Likely Player of the Match")

        teams_list = sorted(teams)
        all_venues = sorted(df["venue"].dropna().unique()) if "venue" in df.columns else []
        toss_choices = sorted(df["toss_decision"].dropna().unique()) if "toss_decision" in df.columns else ["bat", "field"]

        team_a = st.selectbox("Team A (Team1 in ML)", teams_list, index=0, key="pa_team_a")
        team_b = st.selectbox(
            "Team B (Team2 in ML)",
            [t for t in teams_list if t != team_a] if len(teams_list) > 1 else teams_list,
            index=0,
            key="pa_team_b",
        )
        venue_x = st.selectbox("Venue", all_venues, index=0 if all_venues else None, key="pa_venue")
        toss_win = st.selectbox("Toss Winner", [team_a, team_b], index=0, key="pa_toss_win")
        toss_dec = st.selectbox("Toss Decision", toss_choices, index=0, key="pa_toss_dec")

        if st.button("Predict (ML)", key="pa_predict"):
            if team_a == team_b:
                st.warning("Team A and Team B must be different.")
            elif not all_venues:
                st.warning("Venue information missing from dataset.")
            else:
                pred, proba, explanation_lines, best_model_name = ml_predict_winner(
                    team1=team_a,
                    team2=team_b,
                    venue=venue_x,
                    toss_winner=toss_win,
                    toss_decision=toss_dec,
                )
                st.success(f"Predicted Winner: {pred} (P(TeamA win)={proba:.2%}) | {best_model_name}")

                # Likely PoM based on historical matches won by predicted team.
                df_win = df_p.copy()
                if "winner" in df_win.columns:
                    df_win = df_win[df_win["winner"] == pred]
                if "venue" in df_win.columns:
                    df_win = df_win[df_win["venue"] == venue_x]

                if not df_win.empty and "player_of_match" in df_win.columns:
                    likely_pom = df_win["player_of_match"].value_counts().head(5).reset_index()
                    likely_pom.columns = ["Player", "PoM Count"]
                    fig2 = px.bar(
                        likely_pom,
                        x="PoM Count",
                        y="Player",
                        orientation="h",
                        title="Likely Player of the Match (from historical winners)",
                        color_discrete_sequence=["#10B981"],
                    )
                    fig2.update_layout(template="plotly_white")
                    st.plotly_chart(fig2, use_container_width=True)
                if explanation_lines:
                    with st.expander("ML Explanation"):
                        for line in explanation_lines[:5]:
                            st.write(f"- {line}")


# ---------------- SEASON TRENDS ----------------
with tab_season:
    st.subheader("Season Trends")

    if "match_year" in df.columns:
        years = pd.to_numeric(df["match_year"], errors="coerce")
        min_y, max_y = int(years.min()), int(years.max())
        year_range = st.slider(
            "Year range",
            min_value=min_y,
            max_value=max_y,
            value=(min_y, max_y),
            step=1,
            key="season_year_range",
        )
        df_s = df[(pd.to_numeric(df["match_year"], errors="coerce") >= year_range[0]) & (pd.to_numeric(df["match_year"], errors="coerce") <= year_range[1])].copy()
    else:
        df_s = df.copy()

    matches_per_season = df_s.groupby("match_year").size().reset_index(name="matches")
    fig = px.line(matches_per_season, x="match_year", y="matches",
                  markers=True, title="Matches Per Season",
                  color_discrete_sequence=["#2563EB"])
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    try:
        fig2, winners_df = season_winners_trend(df_s)
        if not winners_df.empty and "match_year" in winners_df.columns:
            winners_last_n = st.slider("Show last N seasons in winners trend", min_value=3, max_value=10, value=5, step=1)
            winners_df = winners_df.tail(int(winners_last_n))
            # Rebuild figure based on filtered winners_df for consistency
            fig2 = px.bar(
                winners_df,
                x="match_year",
                y="wins",
                color="winner",
                title="Season-wise Winners Trend (filtered)",
            )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(winners_df)
    except:
        st.warning("Season winners data not available.")

    st.divider()
    st.subheader("ML Prediction: Winner of a Match (latest-season context)")
    teams_list = sorted(teams)
    all_venues = sorted(df["venue"].dropna().unique()) if "venue" in df.columns else []
    toss_choices = sorted(df["toss_decision"].dropna().unique()) if "toss_decision" in df.columns else ["bat", "field"]

    team_a = st.selectbox("Team A (Team1 in ML)", teams_list, index=0, key="season_team_a")
    team_b = st.selectbox(
        "Team B (Team2 in ML)",
        [t for t in teams_list if t != team_a] if len(teams_list) > 1 else teams_list,
        index=0,
        key="season_team_b",
    )
    venue_s = st.selectbox("Venue", all_venues, index=0 if all_venues else None, key="season_venue")
    toss_win = st.selectbox("Toss Winner", [team_a, team_b], index=0, key="season_toss_win")
    toss_dec = st.selectbox("Toss Decision", toss_choices, index=0, key="season_toss_dec")

    if st.button("Predict Winner (ML)", key="season_predict"):
        if not all_venues:
            st.warning("Venue information missing from dataset.")
        elif team_a == team_b:
            st.warning("Team A and Team B must be different.")
        else:
            pred, proba, explanation_lines, best_model_name = ml_predict_winner(
                team1=team_a,
                team2=team_b,
                venue=venue_s,
                toss_winner=toss_win,
                toss_decision=toss_dec,
            )
            st.success(f"Predicted Winner: {pred} (P(TeamA win)={proba:.2%}) | {best_model_name}")
            if explanation_lines:
                with st.expander("ML Explanation"):
                    for line in explanation_lines[:5]:
                        st.write(f"- {line}")


# ---------------- ML PREDICTION LAB ----------------
with tab_ml:
    st.subheader("ML Prediction Lab")
    st.write("This section builds an ML winner predictor using a binary target (Team1 win vs loss). It trains multiple algorithms and compares them.")

    try:
        ml = train_ml_cached(ml_split_mode)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    metrics_df = ml["metrics_df"].copy()
    st.markdown("### Model Performance (Binary Winner Prediction)")
    st.dataframe(metrics_df.style.format({
        "accuracy": "{:.3f}",
        "precision": "{:.3f}",
        "recall": "{:.3f}",
        "f1": "{:.3f}",
        "roc_auc": "{:.3f}",
        "log_loss": "{:.3f}",
    }))

    # Graph 1: Model comparison bar chart
    plot_cols = [c for c in ["f1", "roc_auc", "accuracy"] if c in metrics_df.columns]
    long_metrics = metrics_df.melt(
        id_vars="model",
        value_vars=plot_cols,
        var_name="metric",
        value_name="score",
    )
    fig1 = px.bar(
        long_metrics,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Graph 1: Model Metric Comparison",
    )
    fig1.update_layout(template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    # Graph 2: Confusion matrix heatmap for best model
    cm = ml["confusion_matrix"]
    labels = ml["labels"]
    fig2 = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title=f"Graph 2: Confusion Matrix ({ml['best_model_name']})",
    )
    fig2.update_layout(template="plotly_white", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig2, use_container_width=True)

    # Graph 3: Feature importance (top features from best model)
    fi = ml["feature_importance_df"]
    if not fi.empty:
        fig3 = px.bar(
            fi.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Graph 3: Top Feature Importance",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig3.update_layout(template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Graph 3 skipped: feature importance is unavailable for this model.")

    # Graph 4: Calibration curve (probability vs outcome frequency)
    cal = ml.get("calibration_df", pd.DataFrame())
    if not cal.empty and {"mean_predicted_probability", "fraction_positives"}.issubset(cal.columns):
        fig4 = px.line(
            cal,
            x="mean_predicted_probability",
            y="fraction_positives",
            markers=True,
            title="Graph 4: Calibration Curve (Best Model)",
        )
        fig4.update_layout(template="plotly_white")
        fig4.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line=dict(dash="dash", color="gray"),
            )
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Graph 4 skipped: calibration data not available for this model.")

    # Graph 5: Confidence distribution (probability scores for Team1_win)
    conf = ml.get("confidence_scores", np.array([]))
    if isinstance(conf, np.ndarray) and conf.size > 0:
        fig_c = px.histogram(
            x=conf,
            nbins=20,
            title="Graph 5: Confidence Distribution (P(Team1 win))",
            color_discrete_sequence=["#2563EB"],
        )
        fig_c.update_layout(template="plotly_white")
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Graph 5 skipped: confidence distribution not available.")

    # Graph 6: Predicted outcome share (test set)
    pred_dist = ml.get("test_pred_distribution", pd.DataFrame())
    if not pred_dist.empty and {"result", "pred_count"}.issubset(pred_dist.columns):
        fig_pie = px.pie(
            pred_dist.head(10),
            names="result",
            values="pred_count",
            title="Graph 6: Predicted Outcome Share (Test Set)",
        )
        fig_pie.update_layout(template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Graph 6 skipped: predicted outcome distribution not available.")

    # Graph 7: Learning curve (F1 vs training size)
    lc = ml.get("learning_curve_df", pd.DataFrame())
    if not lc.empty and {"train_size", "train_f1", "val_f1"}.issubset(lc.columns):
        fig5 = px.line(
            lc,
            x="train_size",
            y=["train_f1", "val_f1"],
            markers=True,
            title="Graph 5: Learning Curve (F1)",
        )
        fig5.update_layout(template="plotly_white")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Graph 5 skipped: learning curve data not available.")

    st.markdown("### Multi-Match Winner Predictor")
    st.write("Add multiple match inputs and get predictions for all of them (no static/sample matches).")

    all_teams = sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique())
    all_venues = sorted(df["venue"].dropna().unique()) if "venue" in df.columns else []
    toss_choices = (
        sorted(df["toss_decision"].dropna().unique()) if "toss_decision" in df.columns else ["bat", "field"]
    )

    n_matches = st.number_input(
        "Number of matches to predict",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
    )

    with st.form("multi_predict_form"):
        match_inputs = []
        for i in range(int(n_matches)):
            st.markdown(f"#### Match {i + 1}")
            c1, c2, c3 = st.columns(3)
            with c1:
                t1 = st.selectbox(f"Team 1 (Match {i + 1})", all_teams, index=0, key=f"t1_{i}")
            with c2:
                t2_default = 1 if len(all_teams) > 1 else 0
                t2 = st.selectbox(f"Team 2 (Match {i + 1})", all_teams, index=t2_default, key=f"t2_{i}")
            with c3:
                toss_w = st.selectbox(f"Toss Winner (Match {i + 1})", all_teams, index=0, key=f"tw_{i}")

            c4, c5 = st.columns(2)
            with c4:
                v = st.selectbox(
                    f"Venue (Match {i + 1})",
                    all_venues,
                    index=0 if all_venues else None,
                    key=f"v_{i}",
                )
            with c5:
                toss_dec = st.selectbox(
                    f"Toss Decision (Match {i + 1})",
                    toss_choices,
                    index=0 if toss_choices else None,
                    key=f"td_{i}",
                )

            match_inputs.append(
                {
                    "team1": t1,
                    "team2": t2,
                    "toss_winner": toss_w,
                    "venue": v,
                    "toss_decision": toss_dec,
                }
            )

        submitted = st.form_submit_button("Predict All Winners")

    if submitted:
        results = []
        for i, inp in enumerate(match_inputs):
            team1 = inp["team1"]
            team2 = inp["team2"]
            if team1 == team2:
                results.append(
                    {
                        "Match": i + 1,
                        "Team 1": team1,
                        "Team 2": team2,
                        "Predicted Winner": "Invalid (Team1=Team2)",
                        "Confidence (Team1 win)": None,
                    }
                )
                continue

            raw_row = build_prediction_row(
                raw_df=df,
                team1=team1,
                team2=team2,
                venue=inp["venue"],
                toss_winner=inp["toss_winner"],
                toss_decision=inp["toss_decision"],
            )
            pred, proba = predict_match_winner(
                ml["best_model"],
                raw_row,
                team1_name=team1,
                team2_name=team2,
            )

            results.append(
                {
                    "Match": i + 1,
                    "Team 1": team1,
                    "Team 2": team2,
                    "Venue": inp["venue"],
                    "Toss Winner": inp["toss_winner"],
                    "Toss Decision": inp["toss_decision"],
                    "Predicted Winner": pred,
                    "Confidence (Team1 win)": None if proba is None else float(proba),
                }
            )

        out_df = pd.DataFrame(results)
        if "Confidence (Team1 win)" in out_df.columns:
            out_df["Confidence (Team1 win)"] = out_df["Confidence (Team1 win)"].apply(
                lambda x: None if pd.isna(x) else f"{x:.2%}"
            )

        st.dataframe(out_df)

        st.markdown("### Explanations (Top Factors / Engineered Features)")
        for i in range(len(results)):
            if results[i]["Predicted Winner"].startswith("Invalid"):
                continue
            raw_row = build_prediction_row(
                raw_df=df,
                team1=results[i]["Team 1"],
                team2=results[i]["Team 2"],
                venue=results[i]["Venue"],
                toss_winner=results[i]["Toss Winner"],
                toss_decision=results[i]["Toss Decision"],
            )
            try:
                explanation_lines = explain_prediction_from_global_importance(
                    ml["feature_importance_df"],
                    raw_row,
                )
                with st.expander(f"Match {i + 1} explanation"):
                    for line in explanation_lines[:5]:
                        st.write(f"- {line}")
            except Exception as e:
                with st.expander(f"Match {i + 1} explanation"):
                    st.info(f"Explanation unavailable: {e}")


