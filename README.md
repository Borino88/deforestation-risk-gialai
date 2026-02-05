# Deforestation Risk Prediction (Gia Lai, Vietnam)

This repository provides a reproducible pipeline for 1 km grid-based deforestation risk modeling in Gia Lai Province.

## Inputs 
Upload these CSV exports to your Colab runtime (`/content/`):
- KBang_TRAIN_master_rain_elev_lossyear.csv
- KBang_TEST_master_rain_elev_lossyear.csv
- MangYang_TRAIN_master_rain_elev_lossyear.csv
- MangYang_TEST_master_rain_elev_lossyear.csv

## Reproducible run (Google Colab)
### Step A — Clone the repo + install requirements
Run this in a Colab cell:
```python
!git clone https://github.com/Borino88/deforestation-risk-gialai.git
%cd deforestation-risk-gialai
!pip -q install -r requirements.txt
