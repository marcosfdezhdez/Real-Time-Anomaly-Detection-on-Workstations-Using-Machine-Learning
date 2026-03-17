import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
import joblib

DATASET_FILE = "dataset_labeled.csv"
MODEL_FILE = "logistic_regression_model.pkl"

def main():
    df = pd.read_csv(DATASET_FILE)

    feature_cols = [
        "cpu_percent",
        "ram_percent",
        "disk_read_delta",
        "disk_write_delta",
        "net_sent_delta",
        "net_recv_delta",
    ]

    X = df[feature_cols].fillna(0)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # important because less anomalous than normal samples
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("F1 score (anomaly=1):", f1_score(y_test, y_pred))

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_FILE)
    print(f"\nSaved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
