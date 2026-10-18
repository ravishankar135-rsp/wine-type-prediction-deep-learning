# Wine Type Prediction Using a Neural Network

A beginner-friendly data science project that predicts whether a sample is **red** or **white** wine using physicochemical measurements.

## Project contents
- `notebooks/Wine_Type_Prediction.ipynb`: step-by-step notebook
- `src/train_model.py`: readable training/evaluation script
- `src/download_data.py`: downloads the two public CSV datasets
- `reports/metrics.json`: metrics from the verified local run
- `reports/sample_predictions.csv`: sample predictions
- `requirements.txt`: Python dependencies

## Setup and run
Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/download_data.py
python src/train_model.py
```

macOS/Linux:
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/download_data.py
python src/train_model.py
```

The download step saves `redwinequality.csv` and `whitewinequality.csv` under `data/`. Open the notebook in Google Colab or Jupyter; in Colab, upload both CSVs when prompted.

## Verified local run
Using 6,497 rows (5,197 train / 1,300 test), the current run achieved **99.77% test accuracy**. Confusion matrix (white, red): `[[980, 0], [3, 317]]`. Results may vary across library versions and training settings.

The script creates `models/wine_type_model.joblib`, `reports/metrics.json`, and `reports/sample_predictions.csv`.

## Notes
- This predicts wine type, not wine quality.
- `quality` is included as an input feature because it exists in both source CSVs; remove it from `FEATURE_COLUMNS` to compare results without it.
- This is an educational project, not a guarantee for every future sample.
