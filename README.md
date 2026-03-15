# 🛒 E-Commerce Customer Behavior Prediction using Machine Learning

> An end-to-end data science project covering data generation, cleaning,
> EDA, customer segmentation, purchase prediction, time series analysis,
> sentiment analysis, and a Tkinter GUI application.

---

## 📁 Project Structure

```
ecommerce_behavior_prediction/
│
├── data/
│   └── ecommerce_data.csv          ← auto-generated synthetic dataset
│
├── notebooks/
│   └── ecommerce_behavior_prediction.ipynb   ← master notebook (all modules)
│
├── scripts/
│   ├── generate_dataset.py         ← synthetic data generator
│   ├── preprocessing.py            ← cleaning & feature engineering utilities
│   ├── train_models.py             ← trains & saves ML models
│   └── sentiment_analysis.py       ← VADER / fallback sentiment scorer
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl           ← best model
│   ├── artefacts.pkl               ← encoders, scaler, test split
│   └── feature_importances.csv
│
├── visualizations/
│   ├── monthly_trends.png
│   ├── category_revenue.png
│   ├── demographics.png
│   ├── correlation_heatmap.png
│   ├── elbow_method.png
│   ├── customer_segments.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importances.png
│   ├── time_series.png
│   ├── seasonal_heatmap.png
│   └── sentiment_analysis.png
│
├── gui/
│   └── gui_app.py                  ← Tkinter sentiment predictor app
│
├── docs/
│   └── tableau_guide.md            ← steps to build the Tableau dashboard
│
└── README.md
```

---

## ⚡ Quick Start

### 1 · Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

### 2 · Generate the dataset

```bash
cd ecommerce_behavior_prediction
python scripts/generate_dataset.py
```

This creates `data/ecommerce_data.csv` (~15 000 rows, ~2 000 customers).

### 3 · Run the Jupyter notebook

```bash
jupyter notebook notebooks/ecommerce_behavior_prediction.ipynb
```

Execute cells in order — the notebook is self-contained.

### 4 · Train models via CLI (optional)

```bash
python scripts/train_models.py
```

Saved models appear in `models/`.

### 5 · Launch the GUI

```bash
python gui/gui_app.py
```

Type any customer review and click **Predict Sentiment**.

---

## 🗃️ Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | str | Unique transaction identifier |
| customer_id | str | Unique customer identifier |
| age | int | Customer age |
| gender | str | Male / Female / Other |
| location | str | US city |
| loyalty_tier | str | Bronze / Silver / Gold / Platinum |
| category | str | Product category (10 types) |
| product | str | Product name |
| price | float | Unit price |
| quantity | int | Units purchased |
| total_amount | float | price × quantity |
| rating | float | 1–5 star rating |
| review | str | Raw text review |
| sentiment | str | Positive / Neutral / Negative (ground truth) |
| payment_method | str | Payment type |
| timestamp | datetime | Purchase date-time |
| will_purchase_again | int | Target label (0 / 1) |

---

## 🧪 Modules

### Data Cleaning (`scripts/preprocessing.py`)
- Duplicate removal
- Null imputation (median for `rating`, text fill for `review`)
- Regex-based review text cleaning
- Type conversion & time-feature extraction

### EDA (`notebooks/` → Section 3)
- Monthly revenue & order trends
- Category revenue ranking
- Customer demographics (age histogram, gender pie, loyalty bar)
- Pearson correlation heatmap

### Customer Segmentation (`notebooks/` → Section 4)
- K-Means with k = 4 chosen via elbow method
- PCA 2-D scatter for visual inspection
- Segment profiles (spend, recency, frequency, rating)

### ML Models (`scripts/train_models.py`)
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline; fast to train |
| Decision Tree | Interpretable; max_depth = 8 |
| **Random Forest** | **Best accuracy & F1** |

Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix

### Time Series (`notebooks/` → Section 6)
- Weekly revenue with 4-week rolling average
- Month × Year heatmap to detect seasonality

### Sentiment Analysis (`scripts/sentiment_analysis.py`)
- VADER (`nltk`) for compound scoring
- Graceful fallback to keyword lexicon
- Per-category sentiment breakdown

### GUI (`gui/gui_app.py`)
- Dark-theme Tkinter window
- Text input → real-time sentiment + compound score
- Prediction history (last 5)

---

## 📊 Tableau Dashboard (Guide)

See `docs/tableau_guide.md` for step-by-step instructions to connect the
CSV exports in `visualizations/` and build four worksheets:

1. **Sales Trends** – Line chart from `monthly_trends.png` data
2. **Customer Segments** – Scatter using cluster labels
3. **Sentiment Distribution** – Pie / stacked bar
4. **Product Performance** – Revenue by category bar

---

## 📈 Results Summary

| Module | Result |
|--------|--------|
| Duplicates removed | ~450 rows |
| Customer segments | 4 (High-Value, Occasional, Budget, Churned) |
| Best ML model | Random Forest (~72–75 % accuracy) |
| Top feature | `total_amount`, `rating`, `loyalty_tier` |
| Positive reviews | ~60 % of dataset |
| Seasonal peak | Q4 (Oct–Dec) each year |

---

## 🔮 Future Enhancements

- Real-time pipeline with Apache Kafka + Airflow
- REST API deployment with FastAPI
- Fine-tuned BERT for higher-accuracy sentiment
- Collaborative filtering for product recommendations
- A/B test framework for targeted marketing segments

---

## 👤 Author

Built as a complete end-to-end Data Science portfolio project.  
Dataset is synthetically generated for demonstration purposes.
