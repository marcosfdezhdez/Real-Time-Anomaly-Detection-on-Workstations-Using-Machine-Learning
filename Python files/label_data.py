import pandas as pd

# ========= CONFIGURATION =========
INPUT_FILE = "system_metrics.csv"
OUTPUT_FILE = "dataset_labeled.csv"

ANOMALY_START = "2026-01-12T15:11:18.221386"    #timestamp recorded when recording with monitor.py
ANOMALY_END   = "2026-01-12T15:12:18.221386"    #timestamp recorded when recording with monitor.py
# =================================

def main():
    df = pd.read_csv(INPUT_FILE)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort just in case
    df = df.sort_values("timestamp")

    # Compute deltas for disk and network
    df["disk_read_delta"] = df["disk_read_bytes"].diff().fillna(0)
    df["disk_write_delta"] = df["disk_write_bytes"].diff().fillna(0)
    df["net_sent_delta"] = df["net_sent_bytes"].diff().fillna(0)
    df["net_recv_delta"] = df["net_recv_bytes"].diff().fillna(0)

    # Label anomalies
    start = pd.to_datetime(ANOMALY_START)
    end = pd.to_datetime(ANOMALY_END)

    df["label"] = 0
    df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end), "label"] = 1

    # Select final columns
    df_final = df[
        [
            "timestamp",
            "cpu_percent",
            "ram_percent",
            "disk_read_delta",
            "disk_write_delta",
            "net_sent_delta",
            "net_recv_delta",
            "label",
        ]
    ]

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}")
    print(df_final["label"].value_counts())

if __name__ == "__main__":
    main()
