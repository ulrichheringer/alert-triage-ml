# Alert Triage ML: Log Clustering & Noise Detection

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)  [![MIT License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)  ![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-6b7280?style=for-the-badge)

Groups similar logs with K-Means and classifies alerts that may be noisy.

## Data

Deterministic synthetic data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, log clustering reached a cosine silhouette score of 0.7701 on 175 held-out logs. On a separate 300-alert holdout, noisy-alert classification reached 0.8533 accuracy, 0.8667 F1, and 0.9319 ROC AUC. The noisy-alert labels come from synthetic rules, so these scores do not reflect real incidents.

## Limitations

The generated logs and labels are simplified. Real use would require representative incident and alert history, with noise labels reviewed against actual operator outcomes.
