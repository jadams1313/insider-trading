Parent Paper Implementation

This repository provides the implementation for the Parent Paper project, focused on analyzing company trades through litigation-based classification and financial sentiment modeling. The system integrates a Litigation Classifier with FinBERT to identify and extract relevant companies, then performs trade-level anomaly detection using multiple time-series comparison methods.
Project Overview

The workflow is composed of three main stages:

Company Identification (Litigation Classifier + BERT(uses FinBERT)

Trade Analysis (run.py in trades/ directory)

Anomaly Detection & Comparison (detection.py)

Pipeline Description
1. Litigation Classifier + FinBERT

Directory: litigation/litigation_classifier.py 

This module uses BERT models for Litigation Classifier. 

The classifier identifies companies involved in litigation or legal events.

The output is a curated list of companies for downstream trade analysis. 

WARNING: This code takes hours to run the full dataset. Run a sampled one for assured performance. 

2. Trade Analysis:

Once the target companies are identified, their trading data is analyzed using run.py.
This script retrieves, cleans, and structures historical trading records for each company.

3. Detection:

This stage compares multiple time-series range methods to detect anomalous trading patterns.

Implements an anomalous detection algorithm across different temporal windows.
Evaluates and contrasts various range-detection approaches to highlight irregular trade behaviors.

Structure 

insider-trading/
│
├── litigation/
│   ├── litigation_classifier.py
|   └── data
│
├── trades/
│   ├── run.py
│   └── data/
│
├── dection
│   ├── data/
│   ├── output/
│   └──detection.py
└── README.md
