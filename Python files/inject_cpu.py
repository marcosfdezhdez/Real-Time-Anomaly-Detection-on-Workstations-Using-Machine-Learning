import time
import math
import multiprocessing as mp #to generate CPU load using multiple processes, so that the anomaly affects the overall CPU usage.

DURATION_SECONDS = 60
NUM_WORKERS = max(1, mp.cpu_count() - 1)

def burn_cpu(stop_time):    #consums CPU until time runs out, in this case 60 seconds since the start of the injection
    x = 0.0001
    while time.time() < stop_time:
        x = math.sqrt(x) * math.sqrt(x + 1.2345)
        if x > 1e6:
            x = 0.0001

def main():
    stop_time = time.time() + DURATION_SECONDS
    procs = []
    print(f"Starting CPU increment usage for {DURATION_SECONDS}s")

    for _ in range(NUM_WORKERS):    #creation of many processes
        p = mp.Process(target=burn_cpu, args=(stop_time,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("CPU anomaly injection completed, arrivederci")

if __name__ == "__main__":
    main()
