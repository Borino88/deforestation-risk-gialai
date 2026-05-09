# Deforestation Risk Prediction in Vietnam (Gia Lai Pilot)

[![DOI](https://zenodo.org/badge/1150227675.svg)](https://doi.org/10.5281/zenodo.18491114)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Borino88/deforestation-risk-gialai/blob/master/notebooks/run_in_colab.ipynb)

This repository contains the interpretable machine learning pipeline for deforestation risk prediction at 1 km resolution, as presented in the research paper: **"An Interpretable Machine Learning Pipeline for Deforestation Risk Prediction in Vietnam at 1 km Resolution"**.

## Research Overview
The project focuses on K’Bang and Mang Yang districts in Gia Lai Province, Vietnam. It integrates multi-source datasets (Hansen GFC, Sentinel-2, CHIRPS, SRTM) to predict forest loss risk using Logistic Regression and Random Forest models.

## Contributors
| Name | Role | GitHub |
| :--- | :--- | :--- |
| **Pham Duy Long** | First Author | [@pdlong-09](https://github.com/pdlong-09) |
| **Nguyen Vu Huy** | Co-Author | [@bob-the-guy](https://github.com/bob-the-guy) |
| **Nguyen Duc Anh** | Co-Author | [@nnda29](https://github.com/nnda29) |
| **Do Nhat Quang** | Co-Author | [@quang0214adv-cpu](https://github.com/quang0214adv-cpu) |

For a full list of contributors and affiliations, see [CONTRIBUTORS.md](./CONTRIBUTORS.md).

## Repository Structure
- `data/`: Input CSV files (Master tables for training/testing).
- `src/`: Core Python pipeline script (`pipeline.py`).
- `results/`: Sample model outputs, figures (ROC, Feature Importance), and performance tables.
- `notebooks/`: Jupyter/Colab notebooks for reproducible runs.

## How to Run

### 1. Local Environment
Clone the repository and install dependencies:
```bash
git clone https://github.com/Borino88/deforestation-risk-gialai.git
cd deforestation-risk-gialai
pip install -r requirements.txt
```

Run the pipeline:
```python
from src.pipeline import run_pipeline

run_pipeline(
    kbang_train_path="data/KBang_TRAIN_master_rain_elev_lossyear.csv",
    kbang_test_path="data/KBang_TEST_master_rain_elev_lossyear.csv",
    mang_train_path="data/MangYang_TRAIN_master_rain_elev_lossyear.csv",
    mang_test_path="data/MangYang_TEST_master_rain_elev_lossyear.csv",
    out_dir="outputs"
)
```

### 2. Google Colab
Click the "Open in Colab" badge above or open `notebooks/run_in_colab.ipynb` directly in Colab. The notebook is configured to automatically download the data from this repository.

## Outputs
The pipeline generates:
- `QC_summary.csv`: Quality control check of input data.
- `Table3_model_performance.csv`: AUC, Precision, Recall, and F1 metrics.
- `Figure_RF_importance.png`: Top risk drivers identified by Random Forest.
- `Model_predictions_and_warning_zones.csv`: Spatial risk scores for Gia Lai.
