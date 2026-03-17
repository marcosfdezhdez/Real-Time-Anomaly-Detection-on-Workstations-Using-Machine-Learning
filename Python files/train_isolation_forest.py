import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import IsolationForest
import joblib

DATASET_FILE = "dataset_labeled.csv"
MODEL_FILE = "isolation_forest_model.pkl"

def main():
    df = pd.read_csv(DATASET_FILE)

    # Features (X) and Labels (y)
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

    # Split train/test (maintaining proportion of anomalies)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Isolation Forest is "unsupervised", but I provide the expected contamination level
    contamination = y_train.mean()  # proportion of anomalies in the training set
    model = IsolationForest(
        n_estimators=200,
        random_state=42,
        contamination=contamination
    )

    model.fit(X_train)

    # IsolationForest returns: 1 for normal, -1 for anomaly
    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("F1 score (anomaly=1):", f1_score(y_test, y_pred))

    # Save the trained model
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_FILE)
    print(f"\nSaved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
