import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


TEAM_NAME_MAPPING = {
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors",
    "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",
}


def load_data(path="data/matches.csv"):
    """Load raw CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data preprocessing steps:
    - Handle missing values (drop rows missing critical fields)
    - Convert `date` to datetime
    - Remove unused columns (id, umpires)
    - Standardize team names using `TEAM_NAME_MAPPING`
    - Create `match_year`, `win_margin`, `toss_win_match_win`

    Returns cleaned DataFrame.
    """
    df = df.copy()

    # Basic drop of rows missing winners or teams
    critical_cols = ["team1", "team2", "date", "winner"]
    df.dropna(subset=critical_cols, inplace=True)

    # Convert date to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # If season column exists, use it; otherwise extract year from date
        if "season" in df.columns:
            df["match_year"] = df["season"].astype(int)
        else:
            df["match_year"] = df["date"].dt.year

    # Remove some unused columns if present
    for col in ["id", "umpire1", "umpire2", "umpire3"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Standardize team names
    for c in ["team1", "team2", "toss_winner", "winner"]:
        if c in df.columns:
            df[c] = df[c].replace(TEAM_NAME_MAPPING)

    # Create win_margin: prefer win_by_runs else win_by_wickets
    if "win_by_runs" in df.columns and "win_by_wickets" in df.columns:
        df["win_margin"] = df.apply(
            lambda r: int(r["win_by_runs"]) if r.get("win_by_runs", 0) > 0 else int(r.get("win_by_wickets", 0)),
            axis=1,
        )
    else:
        df["win_margin"] = np.nan

    # toss winner also match winner flag
    if "toss_winner" in df.columns and "winner" in df.columns:
        df["toss_win_match_win"] = (df["toss_winner"] == df["winner"]).astype(int)

    # Fill NA for player_of_match if exists
    if "player_of_match" in df.columns:
        df["player_of_match"] = df["player_of_match"].fillna("Unknown")

    return df


# --- Analysis helpers and plotting functions ---
def total_matches_by_team(df: pd.DataFrame) -> pd.DataFrame:
    # Count occurrences where a team appears as team1 or team2
    teams = pd.concat([df["team1"], df["team2"]], axis=0)
    matches = teams.value_counts().rename_axis("team").reset_index(name="matches")
    return matches


def total_wins_by_team(df: pd.DataFrame) -> pd.DataFrame:
    wins = df["winner"].value_counts().rename_axis("team").reset_index(name="wins")
    return wins


def win_percentage(df: pd.DataFrame) -> pd.DataFrame:
    matches = total_matches_by_team(df)
    wins = total_wins_by_team(df)
    merged = matches.merge(wins, on="team", how="left").fillna(0)
    merged["win_percentage"] = (merged["wins"] / merged["matches"]) * 100
    merged.sort_values("win_percentage", ascending=False, inplace=True)
    return merged


def plot_matches_bar(df: pd.DataFrame):
    matches = total_matches_by_team(df)
    fig = px.bar(matches, x="team", y="matches", title="Total Matches Played by Each Team")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_wins_bar(df: pd.DataFrame):
    wins = total_wins_by_team(df)
    fig = px.bar(wins, x="team", y="wins", title="Total Wins by Each Team")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_win_percentage_pie(df: pd.DataFrame):
    wp = win_percentage(df)
    fig = px.pie(wp, names="team", values="win_percentage", title="Win Percentage of Each Team")
    return fig


def plot_toss_impact(df: pd.DataFrame):
    if "toss_win_match_win" not in df.columns:
        return go.Figure()
    agg = df.groupby("toss_win_match_win").size().reset_index(name="count")
    agg["label"] = agg["toss_win_match_win"].map({1: "Toss Winner also Match Winner", 0: "Toss Winner Lost Match"})
    fig = px.pie(agg, names="label", values="count", title="Toss Decision Impact (Toss win vs Match win)")
    return fig


def plot_venue_advantage(df: pd.DataFrame, top_n_venues=6):
    # For each venue show the top winning teams
    if "venue" not in df.columns:
        return go.Figure()
    venue_counts = df["venue"].value_counts().nlargest(top_n_venues).index
    sub = df[df["venue"].isin(venue_counts)]
    grp = sub.groupby(["venue", "winner"]).size().reset_index(name="wins")
    fig = px.bar(grp, x="venue", y="wins", color="winner", title="Venue Advantage: Wins by Team at Top Venues")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def player_of_match_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "player_of_match" not in df.columns:
        return pd.DataFrame(columns=["player", "awards"]) 
    pom = df["player_of_match"].value_counts().rename_axis("player").reset_index(name="awards")
    return pom


def plot_player_of_match(df: pd.DataFrame, top_n=15):
    pom = player_of_match_counts(df).head(top_n)
    fig = px.bar(pom, x="player", y="awards", title=f"Top {top_n} Player of the Match Awards")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def season_winners_trend(df: pd.DataFrame):
    if "match_year" in df.columns:
        # For each year find the winner of the final (latest match or winner with most wins?)
        # Simpler: choose the team with max wins that season
        grp = df.groupby(["match_year", "winner"]).size().reset_index(name="wins")
        idx = grp.groupby("match_year")["wins"].idxmax()
        winners = grp.loc[idx].sort_values("match_year")
        fig = px.bar(winners, x="match_year", y="wins", color="winner", title="Season-wise Winners Trend (top team by wins per season)")
        return fig, winners
    return go.Figure(), pd.DataFrame()


def top_venues(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    if "venue" not in df.columns:
        return pd.DataFrame()
    v = df["venue"].value_counts().rename_axis("venue").reset_index(name="count").head(top_n)
    return v


def plot_top_venues(df: pd.DataFrame, top_n=10):
    v = top_venues(df, top_n)
    fig = px.bar(v, x="venue", y="count", title=f"Top {top_n} Venues Hosting Matches")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


