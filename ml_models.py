import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


def _safe_split(df: pd.DataFrame, split_mode: str = "latest_season"):
    """
    Time-aware split: train on earlier seasons, test on latest season.
    Falls back to a simple split when match_year is unavailable.
    """
    if "match_year" in df.columns:
        years = pd.to_numeric(df["match_year"], errors="coerce")
        valid_years = years.dropna().astype(int)
        uniq = np.sort(valid_years.unique())
        if len(uniq) > 1:
            if split_mode == "last_two_seasons" and len(uniq) >= 3:
                test_years = set(uniq[-2:])
                train_df = df[~years.isin(test_years)].copy()
                test_df = df[years.isin(test_years)].copy()
            else:
                split_year = int(uniq[-1])
                train_df = df[years < split_year].copy()
                test_df = df[years == split_year].copy()
            if len(train_df) > 0 and len(test_df) > 0:
                return train_df, test_df

    # fallback split
    cut = int(len(df) * 0.8)
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()
    return train_df, test_df


def _make_preprocessor(categorical_features):
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        ]
    )


def _build_models(preprocessor):
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "SVM (RBF)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
    }


def _tune_tree_models(models, X_train, y_train):
    tuned = {}
    for name, pipe in models.items():
        if name not in {"Random Forest", "Gradient Boosting"}:
            tuned[name] = pipe
            continue

        if name == "Random Forest":
            params = {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_split": [2, 5, 10],
            }
        else:
            params = {
                "model__n_estimators": [80, 120, 160],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            }

        search = RandomizedSearchCV(
            pipe,
            param_distributions=params,
            n_iter=6,
            cv=3,
            scoring="f1",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
    return tuned


def _extract_feature_importance(trained_pipeline, top_n=15):
    """
    Return feature importance DataFrame for the final estimator if available.
    """
    model = trained_pipeline.named_steps["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]

    # Get OHE feature names
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        # multiclass -> average absolute coefficient across classes
        if coef.ndim > 1:
            values = np.mean(np.abs(coef), axis=0)
        else:
            values = np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    imp = pd.DataFrame({"feature": feature_names, "importance": values})
    imp = imp.sort_values("importance", ascending=False).head(top_n)
    return imp


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-safe features based only on previous matches.
    """
    work = df.copy()
    if "date" in work.columns:
        work = work.sort_values("date").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    required = ["team1", "team2", "winner"]
    for c in required:
        if c not in work.columns:
            raise ValueError(f"Missing required column: {c}")

    team_matches = {}
    team_wins = {}
    h2h_total = {}
    h2h_team1_wins = {}
    venue_total = {}
    venue_team_wins = {}

    records = []
    for _, r in work.iterrows():
        t1 = str(r["team1"])
        t2 = str(r["team2"])
        winner = str(r["winner"])
        venue = str(r.get("venue", "Unknown Venue"))
        pair_key = "||".join(sorted([t1, t2]))

        t1_total = team_matches.get(t1, 0)
        t1_win = team_wins.get(t1, 0)
        t2_total = team_matches.get(t2, 0)
        t2_win = team_wins.get(t2, 0)
        h_total = h2h_total.get(pair_key, 0)
        h_t1 = h2h_team1_wins.get((pair_key, t1), 0)
        v_total = venue_total.get((venue, t1), 0)
        v_wins = venue_team_wins.get((venue, t1), 0)

        rec = {
            "team1": t1,
            "team2": t2,
            "venue": venue,
            "toss_winner": str(r.get("toss_winner", "Unknown Toss Winner")),
            "toss_decision": str(r.get("toss_decision", "Unknown Toss Decision")),
            "match_year": int(r.get("match_year", 0)),
            "team1_form": (t1_win / t1_total) if t1_total > 0 else 0.5,
            "team2_form": (t2_win / t2_total) if t2_total > 0 else 0.5,
            "team1_vs_team2_h2h": (h_t1 / h_total) if h_total > 0 else 0.5,
            "team1_venue_win_rate": (v_wins / v_total) if v_total > 0 else 0.5,
            "team1_win": int(winner == t1),
        }
        records.append(rec)

        # update state after generating features
        team_matches[t1] = team_matches.get(t1, 0) + 1
        team_matches[t2] = team_matches.get(t2, 0) + 1
        team_wins[winner] = team_wins.get(winner, 0) + 1

        h2h_total[pair_key] = h2h_total.get(pair_key, 0) + 1
        if winner == t1:
            h2h_team1_wins[(pair_key, t1)] = h2h_team1_wins.get((pair_key, t1), 0) + 1

        venue_total[(venue, t1)] = venue_total.get((venue, t1), 0) + 1
        if winner == t1:
            venue_team_wins[(venue, t1)] = venue_team_wins.get((venue, t1), 0) + 1

    out = pd.DataFrame(records)
    out["match_year"] = pd.to_numeric(out["match_year"], errors="coerce").fillna(0).astype(int)
    return out


def train_and_evaluate_models(df: pd.DataFrame, split_mode: str = "latest_season"):
    """
    Train 3 models for winner prediction and return metrics + artifacts.
    """
    model_df = _engineer_features(df)
    categorical_features = ["team1", "team2", "venue", "toss_winner", "toss_decision"]
    numeric_features = ["match_year", "team1_form", "team2_form", "team1_vs_team2_h2h", "team1_venue_win_rate"]
    feature_cols = categorical_features + numeric_features

    train_df, test_df = _safe_split(model_df, split_mode=split_mode)
    X_train, y_train = train_df[feature_cols], train_df["team1_win"]
    X_test, y_test = test_df[feature_cols], test_df["team1_win"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ]
    )
    models = _build_models(preprocessor)
    models = _tune_tree_models(models, X_train, y_train)

    rows = []
    trained = {}
    preds = {}
    probas = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        trained[name] = pipe
        preds[name] = y_pred
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            probas[name] = proba
            row["roc_auc"] = roc_auc_score(y_test, proba)
            row["log_loss"] = log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6))
        else:
            row["roc_auc"] = np.nan
            row["log_loss"] = np.nan
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained[best_model_name]
    best_pred = preds[best_model_name]

    labels = np.array([0, 1])
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    feature_importance_df = _extract_feature_importance(best_model)

    # Confidence scores for histogram.
    confidence_scores = np.array([])
    calibration_df = pd.DataFrame(columns=["mean_predicted_probability", "fraction_positives"])
    if best_model_name in probas:
        confidence_scores = probas[best_model_name]
        frac_pos, mean_pred = calibration_curve(y_test, confidence_scores, n_bins=10, strategy="uniform")
        calibration_df = pd.DataFrame(
            {"mean_predicted_probability": mean_pred, "fraction_positives": frac_pos}
        )

    test_pred_distribution = pd.Series(best_pred).map({1: "Team1 Win", 0: "Team1 Loss"}).value_counts().reset_index()
    test_pred_distribution.columns = ["result", "pred_count"]

    # Learning curve (F1)
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=3, scoring="f1", n_jobs=-1, train_sizes=np.linspace(0.3, 1.0, 6)
    )
    learning_curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_f1": train_scores.mean(axis=1),
            "val_f1": val_scores.mean(axis=1),
        }
    )

    return {
        "feature_cols": feature_cols,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "labels": labels,
        "confusion_matrix": cm,
        "feature_importance_df": feature_importance_df,
        "confidence_scores": confidence_scores,
        "calibration_df": calibration_df,
        "learning_curve_df": learning_curve_df,
        "test_pred_distribution": test_pred_distribution,
        "model_predictions": preds,
        "split_mode": split_mode,
    }


def predict_match_winner(best_model, input_row: pd.DataFrame, team1_name: str, team2_name: str):
    pred = int(best_model.predict(input_row)[0])
    proba = None
    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(input_row)[0][1]
        proba = float(prob)
    winner = team1_name if pred == 1 else team2_name
    return winner, proba


def build_prediction_row(
    raw_df: pd.DataFrame,
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
):
    """
    Build engineered single-row features using historical stats before match_year.
    """
    if "match_year" not in raw_df.columns:
        raise ValueError("match_year column is required in raw_df for engineered features.")

    # Since the UI no longer asks the user for match_year, we default to
    # the latest season in the dataset (and compute features using history before it).
    target_year = int(pd.to_numeric(raw_df["match_year"], errors="coerce").max())

    hist = raw_df.copy()
    if "match_year" in hist.columns:
        hist = hist[pd.to_numeric(hist["match_year"], errors="coerce") < target_year].copy()

    t1_total = ((hist["team1"] == team1) | (hist["team2"] == team1)).sum()
    t1_win = (hist["winner"] == team1).sum()
    t2_total = ((hist["team1"] == team2) | (hist["team2"] == team2)).sum()
    t2_win = (hist["winner"] == team2).sum()
    h2h_total = (
        ((hist["team1"] == team1) & (hist["team2"] == team2))
        | ((hist["team1"] == team2) & (hist["team2"] == team1))
    ).sum()
    h2h_t1_win = (((hist["winner"] == team1)) & (
        ((hist["team1"] == team1) & (hist["team2"] == team2))
        | ((hist["team1"] == team2) & (hist["team2"] == team1))
    )).sum()
    venue_total = (((hist["venue"] == venue)) & (((hist["team1"] == team1) | (hist["team2"] == team1)))).sum()
    venue_t1_win = (((hist["venue"] == venue) & (hist["winner"] == team1))).sum()

    row = pd.DataFrame(
        [
            {
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "match_year": target_year,
                "team1_form": (t1_win / t1_total) if t1_total > 0 else 0.5,
                "team2_form": (t2_win / t2_total) if t2_total > 0 else 0.5,
                "team1_vs_team2_h2h": (h2h_t1_win / h2h_total) if h2h_total > 0 else 0.5,
                "team1_venue_win_rate": (venue_t1_win / venue_total) if venue_total > 0 else 0.5,
            }
        ]
    )
    return row


def explain_prediction_from_global_importance(fi_df: pd.DataFrame, row_df: pd.DataFrame):
    # For models that don't expose feature importances/coefs (e.g., KNN, RBF SVM),
    # fall back to showing the engineered feature values used for this prediction.
    if fi_df.empty:
        fallback_numeric = ["team1_form", "team2_form", "team1_vs_team2_h2h", "team1_venue_win_rate", "match_year"]
        lines = []
        for col in fallback_numeric:
            if col in row_df.columns:
                try:
                    lines.append(f"{col}: {float(row_df.iloc[0][col]):.3f}")
                except Exception:
                    lines.append(f"{col}: {row_df.iloc[0][col]}")
        if not lines:
            return ["Model explanation unavailable for this algorithm."]
        return lines
    lines = []
    for _, r in fi_df.head(5).iterrows():
        f = str(r["feature"])
        if "num__" in f:
            base = f.replace("num__", "")
            if base in row_df.columns:
                lines.append(f"{base}: {float(row_df.iloc[0][base]):.3f}")
        elif "cat__" in f:
            # example: cat__team1_Mumbai Indians
            token = f.split("cat__", 1)[-1]
            parts = token.split("_", 1)
            if len(parts) == 2:
                col, val = parts
                if col in row_df.columns and str(row_df.iloc[0][col]) == val:
                    lines.append(f"{col}: {val}")
    if not lines:
        lines = ["Top global features are available in the Feature Importance graph."]
    return lines


def save_model_bundle(train_output: dict, path: str):
    bundle = {
        "best_model_name": train_output.get("best_model_name"),
        "best_model": train_output.get("best_model"),
        "feature_cols": train_output.get("feature_cols"),
        "categorical_features": train_output.get("categorical_features"),
        "numeric_features": train_output.get("numeric_features"),
        "metrics_df": train_output.get("metrics_df"),
        "feature_importance_df": train_output.get("feature_importance_df"),
        "split_mode": train_output.get("split_mode", "latest_season"),
    }
    joblib.dump(bundle, path)


def load_model_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if "best_model" not in bundle or "feature_cols" not in bundle:
        raise ValueError("Invalid model bundle file.")
    return bundle
