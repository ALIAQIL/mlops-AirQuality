# mlops-AirQuality

DVC data versioning pipeline for UCI Air Quality dataset - 4-stage ML pipeline using Git + DVC.

## Dataset

- **Source**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/static/public/360/air+quality.zip)
- **Samples**: 9,358 hourly air quality measurements (March 2004 – February 2005)
- **Missing values**: Encoded as -200 (replaced with NaN)
- **Target**: Binary classification - high_pollution = 1 if CO(GT) >= 2.0 mg/m³, else 0

## Pipeline Stages

```
data/raw/AirQualityUCI.csv.dvc
            │
            ▼
    ┌─────────────┐
    │   prepare   │ → data/processed/{train,val,test}.csv
    └─────────────┘
            │
            ▼
    ┌───────────────┐
    │   featurize   │ → data/features/{train,val,test}_feat.csv
    └───────────────┘
        │       │
        ▼       ▼
    ┌───────┐ ┌─────────┐
    │ train │ │evaluate │ → models/model.pkl, metrics/
    └───────┘ └─────────┘
```

### Stage 1: prepare.py
- Load raw data, replace -200 with NaN
- Split into train/val/test (70%/10%/20%)
- Create binary target: high_pollution

### Stage 2: featurize.py
- Extract time features (hour, dayofweek, month, is_weekend, is_rushhour)
- Normalize features (min-max scaling)
- Handle missing values with median imputation

### Stage 3: train.py
- RandomForestClassifier (n_estimators=200, max_depth=8)
- Balanced class weights
- Output: models/model.pkl

### Stage 4: evaluate.py
- Evaluate on test set
- Output: metrics/scores.json, metrics/report.json

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.9317 |
| Test F1 | 0.9268 |
| Test ROC-AUC | 0.9822 |
| Threshold OK | True |

## Commands

```bash
# Run pipeline
dvc repro

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff HEAD

# Push to remote
dvc push

# Visualize DAG
dvc dag
```

## Project Structure

```
mlops/
├── data/
│   ├── raw/              # Raw data (DVC tracked)
│   ├── processed/         # Train/val/test splits
│   └── features/         # Featurized data
├── models/               # Trained models
├── metrics/              # Evaluation metrics
├── src/                  # Pipeline scripts
│   ├── prepare.py
│   ├── featurize.py
│   ├── train.py
│   └── evaluate.py
├── params.yaml           # Configuration parameters
├── dvc.yaml              # Pipeline definition
├── dvc.lock              # Lock file (hashes)
└── .dvc/                 # DVC cache
```

## Experiments

- **Experiment 1**: n_estimators=100 → val_accuracy=0.9254
- **Experiment 2**: n_estimators=200 → val_accuracy=0.9213 (test_accuracy=0.9317)

## Tech Stack

- Python 3.12
- DVC 3.66.1
- pandas, scikit-learn
- Git
