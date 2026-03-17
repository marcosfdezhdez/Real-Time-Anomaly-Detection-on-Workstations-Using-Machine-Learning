# Real-Time-Anomaly-Detection-on-Workstations-Using-Machine-Learning

## Project Overview
This project presents a real-time anomaly detection system for personal computers using machine learning. It monitors key system metrics such as CPU, memory, and I/O activity to identify unusual behavior. By leveraging supervised learning techniques, the system is able to detect and report anomalies as they occur.

## Repository Contents
In the repository you can find the folder with the python files, the labeled dataset file which is given to train the anomaly detector without requiring your own data, and a project report which deeply analyses the project, including an introduction to the problem, system monitoring design, dataset creation with anomaly injection, analysis and evaluation of machine learning models, and the final real-time anomaly detection system integration.

## Running instructions: 
In order to run the project correctly, the following order must be followed:
To generate your own dataset:
1. monitor.py
2. inject_cpu.py
3. label_data.py
To train the anomaly detector using the provided dataset:
4. train_logistic.py (selected model) or train_isolation_forest.py
(if Isolation Forest is selected, the detector must load the corresponding pre-trained model)
To run real-time anomaly detection:
5. runtime_detector.py
