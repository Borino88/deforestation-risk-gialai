# An Interpretable Machine Learning Pipeline for Deforestation Risk Prediction in Vietnam at 1 km Resolution

[![CI Pipeline](https://github.com/Borino88/deforestation-risk-vietnam/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Borino88/deforestation-risk-vietnam/actions/workflows/docker-publish.yml)
[![Docker Hub](https://img.shields.io/docker/v/borino88/deforestation-risk-vietnam?label=docker&logo=docker)](https://hub.docker.com/r/borino88/deforestation-risk-vietnam)
[![DOI](https://img.shields.io/badge/DOI-pending_Zenodo_archive-lightgrey)](https://doi.org/10.5281/zenodo.18491114)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Borino88/deforestation-risk-vietnam/blob/main/notebooks/run_in_colab.ipynb)

This repository provides the reproducible code, data architecture, and documentation for the research paper: **"An Interpretable Machine Learning Pipeline for Deforestation Risk Prediction in Vietnam at 1 km Resolution."** 

## Research Summary
Deforestation remains a critical environmental challenge in Vietnam. This project presents a proactive monitoring framework that predicts forest loss risk at a 1 km grid-cell resolution. By integrating multi-source satellite data with interpretable machine learning models, the pipeline identifies high-pressure zones before loss occurs, supporting more effective conservation prioritization.

---

## 🗺️ Architecture & Processing Pipeline

```text
+-------------------------------------------------------------------------------+
|                        MULTI-SOURCE SATELLITE INGESTION                       |
|   [Hansen Global Forest]   [Sentinel-2 NDVI/NBR]   [NASA SRTM Topography]   |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       SPATIAL PREPROCESSING & GRIDDING                        |
|       1 km Grid-Cell Alignment -> Feature Scaling (StandardScaler)          |
+-------------------------------------------------------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
+----------------------------------+        +----------------------------------+
|   LOGISTIC REGRESSION MODEL      |        |      RANDOM FOREST CLASSIFIER    |
|   - Coefficient Interpretability |        |      - Non-Linear Interactions   |
|   - Linear Risk Driver Profiling |        |      - High Discriminative (AUC) |
+----------------------------------+        +----------------------------------+
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                     EVALUATION & EARLY-WARNING ARTIFACTS                      |
|       ROC/AUC Analysis -> Capture Rates -> Warning Zone Geo-CSV Export       |
+-------------------------------------------------------------------------------+
```

---

## Study Area
The pilot implementation focuses on two ecologically significant districts in **Gia Lai Province, Vietnam**:
- **K’Bang District**
- **Mang Yang District**

## Data Sources
The pipeline leverages publicly available environmental and infrastructure datasets:
- **Hansen Global Forest Change (UMD/GEE):** Historical forest cover and loss labels (2001–2023).
- **Global Forest Watch (GFW):** Near-real-time integrated alerts.
- **Sentinel-2 (ESA):** Multispectral imagery for vegetation indices (NDVI, NBR).
- **SRTM (NASA):** Topographic features including elevation and slope.
- **OpenStreetMap (OSM):** Human-accessibility features (distance to roads).

## Methodology Overview
We evaluate and compare three distinct modeling approaches:
1. **Baseline Risk Score:** A proximity-weighted heuristic for initial benchmarking.
2. **Logistic Regression:** Used for scientific interpretability of risk drivers (coefficients).
3. **Random Forest:** Capable of capturing complex, non-linear interactions for high discriminative performance (AUC = 0.89).

## Repository Structure
```text
├── data/                   # Master CSV tables for training and evaluation
├── notebooks/              # Google Colab one-click reproduction workflow
├── src/                    # Core Python pipeline scripts
├── results/                # Reference model outputs and research figures
├── participation_evidence/ # Internal records of research team contributions
└── requirements.txt        # Python package dependencies
```

## Reproducible Run (One-Click)
The easiest way to reproduce the research findings is through Google Colab:
1. Open the [notebooks/run_in_colab.ipynb](https://colab.research.google.com/github/Borino88/deforestation-risk-vietnam/blob/main/notebooks/run_in_colab.ipynb).
2. Follow the "Run All" command. The notebook automatically synchronizes with this repository, installs dependencies, and executes the full model pipeline.

## Docker Quick-Start (Recommended)
Run the complete scientific pipeline instantly using the prebuilt multi-stage Docker container:
```bash
# Pull and execute the hardened non-root container pipeline
docker run --rm -v $(pwd)/outputs:/app/outputs borino88/deforestation-risk-vietnam:latest
```

## Local Installation
To run the pipeline in a local Python environment:
```bash
# Clone the repository
git clone https://github.com/Borino88/deforestation-risk-vietnam.git
cd deforestation-risk-vietnam

# Install dependencies
pip install -r requirements.txt

# Execute the pipeline
python -c "from src.pipeline import run_pipeline; run_pipeline('data/KBang_TRAIN_master_rain_elev_lossyear.csv', 'data/KBang_TEST_master_rain_elev_lossyear.csv', 'data/MangYang_TRAIN_master_rain_elev_lossyear.csv', 'data/MangYang_TEST_master_rain_elev_lossyear.csv', 'outputs')"
```

## Expected Outputs
The pipeline generates a standardized set of research outputs in the `outputs/` directory:
- **Spatial Predictions:** `Model_predictions_and_warning_zones.csv` (1 km resolution risk scores).
- **Validation Figures:** ROC curves and Feature Importance bar charts.
- **Performance Tables:** Summaries of AUC, AP, and F1 scores across multiple scenarios.

## Research Team & Contributions
This project was developed by a team of student researchers with supervision from FPT University:
- **Pham Duy Long:** Project coordination and lead author.
- **Nguyen Vu Huy:** Data preparation and reproducibility engineering.
- **Nguyen Duc Anh:** Data validation and metric evaluation.
- **Do Nhat Quang:** Geospatial visualization and figure organization.
- **Tran Dang An:** Methodology validation and project supervision.
- **Dam Anh Thu:** Statistical data analysis and validation support.

For detailed contribution records, see [CONTRIBUTORS.md](./CONTRIBUTORS.md).

## 🤖 Model Card & Specifications
* **Architecture:** Random Forest Classifier (n_estimators=100) & Logistic Regression (StandardScaler normalization).
* **Input Features:** 1 km aggregated raster features (elevation, slope, precipitation, NDVI/NBR vegetation indices, distance to roads).
* **Target Variable:** Binary deforestation indicator derived from Hansen GEE global forest change loss year datasets.
* **Evaluation Metrics:** Evaluated across out-of-sample test splits using ROC-AUC (0.89+), Average Precision (AP), and Top-10% Capture Rate.

## ⚖️ Ethical Use & Governance
This spatial modeling pipeline is developed strictly for **proactive environmental conservation, forest governance, and scientific research**.
* **Intended Use:** Supporting local conservation agencies, park rangers, and environmental policy analysts in prioritizing patrol routes and allocation of conservation resources in high-risk zones.
* **Prohibited Use:** This model must **NOT** be used to justify punitive commercial land seizures, displacement of indigenous forest communities, or unverified regulatory enforcement without on-the-ground human verification.

## ⚠️ Limitations
* **Spatial Resolution:** Predictions are aggregated at a 1 km grid-cell resolution; micro-scale selective logging or sub-hectare degradation may not be detected.
* **Temporal Lag:** Optical satellite indices (Sentinel-2) are subject to cloud-cover interference during monsoon seasons, which can introduce temporal lag in real-time alert validation.
* **Geographic Specificity:** Trained specifically on the topographical and agricultural dynamics of Gia Lai Province; weights must be re-calibrated before transferring to other biomes.

## Citation & Attribution
If you utilize this pipeline, dataset structure, or modeling methodology in your academic or environmental work, please cite using our `CITATION.cff` metadata or standard BibTeX:

```bibtex
@misc{fattahi2024deforestation,
  author       = {Fattahi, Mahdi and Contributors},
  title        = {An Interpretable Machine Learning Pipeline for Deforestation Risk Prediction in Vietnam at 1 km Resolution},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Borino88/deforestation-risk-gialai}}
}
```

## Contact
For technical collaboration, architectural inquiries, or scientific access, contact **a.borino88@gmail.com** or visit [https://fattahi.xyz](https://fattahi.xyz).
