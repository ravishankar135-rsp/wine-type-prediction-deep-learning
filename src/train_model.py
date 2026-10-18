"""Train a beginner-friendly neural network to classify red vs white wine."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality",
]

def load_wine_data():
    """Load both CSVs and add labels: red=1, white=0."""
    red = pd.read_csv(DATA_DIR / "redwinequality.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "whitewinequality.csv", sep=";")
    red["type"] = 1
    white["type"] = 0
    return pd.concat([red, white], ignore_index=True).dropna()

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    wines = load_wine_data()
    X, y = wines[FEATURE_COLUMNS], wines["type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("neural_network", MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            max_iter=300, random_state=42, early_stopping=True
        )),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predicted)
    print(f"Rows used: {len(wines):,}")
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nClassification report (0=white, 1=red):")
    print(classification_report(y_test, predicted,
          target_names=["White wine", "Red wine"], zero_division=0))
    print("Confusion matrix [white, red]:")
    print(confusion_matrix(y_test, predicted))
    output = X_test.copy()
    output["actual_type"] = y_test.map({0: "White wine", 1: "Red wine"})
    output["predicted_type"] = pd.Series(predicted, index=X_test.index).map(
        {0: "White wine", 1: "Red wine"})
    output["red_wine_probability"] = probabilities
    output["correct"] = output["actual_type"] == output["predicted_type"]
    output.sort_index().head(100).to_csv(REPORTS_DIR / "sample_predictions.csv", index=False)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({
        "rows_used": int(len(wines)), "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)), "test_accuracy": float(accuracy),
        "random_state": 42, "target_mapping": {"0": "White wine", "1": "Red wine"}
    }, indent=2), encoding="utf-8")
    joblib.dump(model, MODEL_DIR / "wine_type_model.joblib")
    print("\nSaved reports/sample_predictions.csv and models/wine_type_model.joblib")

if __name__ == "__main__":
    main()
