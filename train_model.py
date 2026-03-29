import os
import pandas as pd

from analysis import preprocess
from ml_models import train_and_evaluate_models, save_model_bundle


def main():
    root = os.path.dirname(__file__)
    data_path = os.path.join(root, "data", "matches.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = preprocess(pd.read_csv(data_path))
    out = train_and_evaluate_models(df)

    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "ipl_model.joblib")

    save_model_bundle(out, model_path)
    print(f"Saved model bundle to: {model_path}")
    print("Best model:", out["best_model_name"])
    cols = [c for c in ["model", "accuracy", "f1", "roc_auc", "log_loss"] if c in out["metrics_df"].columns]
    print(out["metrics_df"][cols].to_string(index=False))


if __name__ == "__main__":
    main()

