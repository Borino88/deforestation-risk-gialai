# Model Card: Gia Lai Deforestation Risk Prediction Model

**Model Name:** GiaLai-ForestRisk-RF-v1.0  
**Lead Developer:** Mahdi Fattahi (`Borino88`) — Senior Full-Stack & Geospatial ML Engineer  
**Date:** July 2026  
**License:** CC BY 4.0 (Model Documentation) / MIT (Code)

---

## 1. Model Overview
The Gia Lai Deforestation Risk Prediction Model is a supervised machine learning classifier built to predict 1 km² grid-level deforestation risk across K'Bang and Mang Yang districts in Gia Lai Province, Central Highlands, Vietnam. It utilizes Random Forest classification trained on multi-spectral remote sensing indices and topographical features.

## 2. Intended Use Cases
* **Primary Use Case:** Proactive forest conservation planning and resource allocation by regional forestry authorities and environmental monitoring organizations.
* **Secondary Use Case:** Methodological benchmark for interpretable geospatial machine learning in tropical forest ecosystems.
* **Out-of-Scope Use Cases:** Not intended for automated legal enforcement, land seizure decisions, or fine-grained individual property surveillance without field validation.

## 3. Model Architecture & Hyperparameters
* **Algorithm:** Random Forest Classifier (`scikit-learn` 1.6.0).
* **Number of Trees (`n_estimators`):** 200 with balanced class weighting (`class_weight='balanced'`).
* **Maximum Depth (`max_depth`):** 15 to prevent overfitting on spatial clusters.
* **Feature Selection:** 8 input features including NDVI change, EVI slope, elevation (DEM), slope angle, distance to roads, distance to forest edge, historical fire density, and precipitation anomalies.

## 4. Evaluation & Performance Metrics
Validated using 5-fold spatial cross-validation (grouping by 10km grid squares to prevent spatial autocorrelation leakage):
* **ROC AUC Score:** `0.892 ± 0.024` across spatial folds.
* **Precision (High Risk Class):** `0.845`
* **Recall / Sensitivity (High Risk Class):** `0.878`
* **F1-Score:** `0.861`

## 5. Limitations & Ethical Considerations
* **Cloud Cover Contamination:** Optical satellite imagery (Sentinel-2) in the Central Highlands is heavily degraded during the monsoon season (May–October).
* **Spatial Resolution:** Predictions are aggregated at 1 km² resolution; localized illegal logging below 100 meters may not be detected.
* **Ethical Protection:** Model outputs must not be used to displace indigenous forest-dependent communities without comprehensive social and environmental impact assessments.
