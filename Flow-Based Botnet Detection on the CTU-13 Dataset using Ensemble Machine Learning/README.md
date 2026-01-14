Flow-Based Botnet Detection on CTU-13 using Ensemble Machine Learning

This repository provides a fully reproducible research pipeline for detecting botnet traffic using flow-level features on the CTU-13 dataset.
We benchmark classical ML baselines and an ensemble-driven approach and provide end-to-end evaluation with ROC curves, false-positive analysis, and feature importance interpretation.

📂 Repository Structure

Flow-Based Botnet Detection on the CTU-13 Dataset using Ensemble Machine Learning/
│
├── src/
│   └── ctu13_baseline.py
├── figures/
│   ├── roc_curves.png
│   └── feature_importance.png
├── paper/
│   └── Flow_Based_Botnet_Detection_Ensemble_CT13_Harshith.pdf
├── requirements.txt
├── .gitignore
└── README.md

📊 Dataset

The CTU-13 dataset contains labeled real botnet traffic mixed with normal background flows.

Due to licensing restrictions, the dataset is not included.

Download from:
👉 https://www.stratosphereips.org/datasets-ctu13

Place the following file inside: data/raw/capture20110810.binetflow

⚙️ Environment Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

▶️ Run Experiment

python3 src/ctu13_baseline.py

📈 Outputs

File
Description
figures/roc_curves.png
ROC comparison of models
figures/feature_importance.png
Random Forest feature importance
outputs/metrics/results.csv
Accuracy, Precision, Recall, F1, AUC


🧪 Models Evaluated

Model
Accuracy
Precision
Recall
F1
ROC-AUC
FPR
Random Forest
0.99977
0.99913
0.98523
0.99213
0.99962
0.000012
Logistic Regression
0.98549
0.00000
0.00000
0.00000
0.97410
0.00000

🔍 Key Findings
	•	Random Forest significantly outperforms Logistic Regression in:
	•	Precision, Recall, F1-score, and ROC-AUC.
	•	Extremely low false positive rate (1.2 × 10⁻⁵) demonstrates strong real-world viability.
	•	Feature importance highlights protocol-level and temporal flow features critical to botnet discrimination.

⸻

📌 Research Contributions
	•	End-to-end reproducible pipeline for CTU-13.
	•	Flow-level ensemble evaluation with quantitative false-positive analysis.
	•	Visual interpretability via feature importance.
	•	Baseline comparison validating ensemble superiority.

⸻

📜 Paper

The accompanying research paper is located in: paper/Flow_Based_Botnet_Detection_Ensemble_CT13_Harshith.pdf

✍️ Author

Harshith Siddartha Kutumbaka
MS Computer Science (Data Science) – UNC Charlotte
GitHub: https://github.com/Siddarthkutumbaka
