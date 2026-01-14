# Flow-Based Botnet Detection on CTU-13 using Ensemble Learning

This repository provides a fully reproducible pipeline for flow-based botnet detection on the CTU-13 dataset using ensemble machine learning models.

## Dataset

Due to licensing restrictions, the CTU-13 dataset is not included.  
Place the file below in `data/raw/`:

- `capture20110810.binetflow`

Dataset source: https://www.stratosphereips.org/datasets-ctu13

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run Expirement

python3 src/ctu13_baseline.py

Outputs 

File
Description
outputs/metrics/results.csv
Model performance metrics
figures/roc_curves.png
ROC curves
figures/feature_importance.png
Feature importance

Results Snapshot

Model
Accuracy
Precision
Recall
F1
ROC-AUC
RandomForest
0.99977
0.99913
0.98523
0.99213
0.99962
LogisticRegression
0.98549
0.0
0.0
0.0
0.97410

