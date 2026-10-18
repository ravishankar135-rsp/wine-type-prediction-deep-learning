# Wine Type Prediction Using a Neural Network

A beginner-friendly data science project that predicts whether a sample is **red** or **white** wine using physicochemical measurements.

## What it does
1. Loads the red and white wine CSV datasets.
2. Labels red wine as `1` and white wine as `0`.
3. Explores alcohol distributions.
4. Splits the data into stratified training and test sets.
5. Scales features and trains a neural network.
6. Reports accuracy, precision, recall, F1-score, and a confusion matrix.
7. Saves sample predictions and a reusable trained model.

## Project structure
```
data/
  redwinequality.csv
  whitewinequality.csv
notebooks/
  Wine_Type_Prediction.ipynb
src/
  train_model.py
reports/
  metrics.json
  sample_predictions.csv
models/  # created by training
requirements.txt
```

## Setup
Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
```
Windows:
```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/train_model.py
```
macOS/Linux:
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train_model.py
```

The two CSV files belong in `data/` and use semicolon separators. Open `notebooks/Wine_Type_Prediction.ipynb` in Google Colab or Jupyter for the step-by-step walkthrough.

## Verified local run
On the current run, the script used 6,497 rows (5,197 train / 1,300 test) and achieved **99.77% test accuracy**. Confusion matrix (white, red): `[[980, 0], [3, 317]]`. Results may vary across library versions and training settings.

## Try a prediction
The training script writes `reports/sample_predictions.csv` with actual and predicted labels, estimated red-wine probability, and whether each prediction was correct. It also saves `models/wine_type_model.joblib`.

## Notes
- The model predicts wine type, not wine quality.
- `quality` is retained as an input feature because it exists in both source CSVs; remove it from `FEATURE_COLUMNS` to compare performance without it.
- This is an educational project, not a guarantee for all future samples.
