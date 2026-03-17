import psutil
import pandas as pd
import time
from datetime import datetime
import os

OUTPUT_FILE = "system_metrics.csv"
INTERVAL = 1  # seconds

def get_metrics():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()

    return {
        "timestamp": datetime.now().isoformat(), #time converted into a string
        "cpu_percent": cpu,
        "ram_percent": ram,
        "disk_read_bytes": disk.read_bytes,
        "disk_write_bytes": disk.write_bytes,
        "net_sent_bytes": net.bytes_sent,
        "net_recv_bytes": net.bytes_recv,
    }

def main():
    file_exists = os.path.isfile(OUTPUT_FILE)

    while True:
        metrics = get_metrics()
        df = pd.DataFrame([metrics])
        df.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)
        file_exists = True
        print(metrics)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
