import psutil
import pandas as pd
import time
from datetime import datetime
import joblib
import os

MODEL_FILE = "logistic_regression_model.pkl"    #I choose the model I want to work with
LOG_FILE = "runtime_log.csv"
INTERVAL = 1  # seconds

def get_metrics():  #same function as the monitor
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu,
        "ram_percent": ram,
        "disk_read_bytes": disk.read_bytes,
        "disk_write_bytes": disk.write_bytes,
        "net_sent_bytes": net.bytes_sent,
        "net_recv_bytes": net.bytes_recv,
    }

def main():
    # Load model and feature list
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    print("Runtime anomaly detector started.")

    prev_metrics = None
    file_exists = os.path.isfile(LOG_FILE)

    while True:
        metrics = get_metrics()

        if prev_metrics is None:
            prev_metrics = metrics
            time.sleep(INTERVAL)
            continue

        # Compute deltas
        row = {
            "cpu_percent": metrics["cpu_percent"],
            "ram_percent": metrics["ram_percent"],
            "disk_read_delta": metrics["disk_read_bytes"] - prev_metrics["disk_read_bytes"],
            "disk_write_delta": metrics["disk_write_bytes"] - prev_metrics["disk_write_bytes"],
            "net_sent_delta": metrics["net_sent_bytes"] - prev_metrics["net_sent_bytes"],
            "net_recv_delta": metrics["net_recv_bytes"] - prev_metrics["net_recv_bytes"],
        }

        X = pd.DataFrame([row])[feature_cols].fillna(0)
        prediction = model.predict(X)[0]

        status = "ANOMALY" if prediction == 1 else "NORMAL"

        log_row = {
            "timestamp": metrics["timestamp"],
            **row,
            "prediction": status
        }

        df_log = pd.DataFrame([log_row])
        df_log.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)
        file_exists = True

        print(f"[{metrics['timestamp']}] → {status}")

        prev_metrics = metrics
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
