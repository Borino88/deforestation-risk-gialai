# Deforestation Risk Prediction (Gia Lai, Vietnam)

This repository provides a reproducible pipeline for 1 km grid-based deforestation risk modeling in Gia Lai Province.

## Inputs
Provide the following CSV exports (not included in the repo):
- KBang_TRAIN_master_rain_elev_lossyear.csv
- KBang_TEST_master_rain_elev_lossyear.csv
- MangYang_TRAIN_master_rain_elev_lossyear.csv
- MangYang_TEST_master_rain_elev_lossyear.csv

## How to run (Google Colab)
1. Open Google Colab
2. Upload the 4 CSVs into `/content/`
3. Run the pipeline script `src/pipeline.py` (see the Colab steps in the project documentation)

## Outputs
- outputs/ALL_master_FINAL.csv
- outputs/QC_summary.csv
- outputs/tables/Table1_datasets.csv
- outputs/tables/Table2_dataset_summary.csv
- outputs/tables/Table3_model_performance.csv
- outputs/figures/*.png
- outputs/predictions/Model_predictions_and_warning_zones.csv

## Run in Google Colab (recommended)
1. Open a new Colab notebook.
2. Run this cell:

```python
!git clone https://github.com/Borino88/deforestation-risk-gialai.git
%cd deforestation-risk-gialai
!pip -q install -r requirements.txt

