# 🚗 Car Market Trends Analysis with Car Dekho Data

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

A complete beginner-friendly **Data Science project** that analyses the Indian used-car market
using the Car Dekho dataset. The project covers the full data science workflow:

> **Data Cleaning → EDA → Visualisation → Machine Learning → Interactive Dashboard**

---

## 🎯 Problem Statement

Used-car buyers and sellers in India lack a simple tool to understand what factors influence
resale prices. This project analyses 301 real listings from Car Dekho to identify market trends
and build a price prediction model.

---

## 🏆 Objectives

1. Clean and inspect the Car Dekho dataset
2. Perform exploratory data analysis with 13 analyses
3. Visualise price trends across fuel type, seller type, transmission, and year
4. Train a regression model to predict the selling price
5. Deploy an interactive Streamlit dashboard

---

## 📊 Dataset Details

| Property | Value |
|----------|-------|
| Source | Car Dekho (Indian used-car platform) |
| Total rows | 301 listings (299 after removing 2 duplicates) |
| Columns | 9 original + 2 derived |
| Missing values | None |
| Year range | 2003 – 2018 |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| Car_Name | Text | Model name |
| Year | Integer | Year of manufacture |
| Selling_Price | Float | Asking price in Rs Lakhs |
| Present_Price | Float | Current showroom price in Rs Lakhs |
| Kms_Driven | Integer | Total kilometres driven |
| Fuel_Type | Category | Petrol / Diesel / CNG |
| Seller_Type | Category | Dealer / Individual |
| Transmission | Category | Manual / Automatic |
| Owner | Integer | Number of previous owners |

---

## ✨ Key Features

- ✅ Full 10-step data cleaning pipeline with flagging (no silent row deletion)
- ✅ 13 EDA analyses with saved PNG charts
- ✅ 4-page interactive Streamlit dashboard
- ✅ Live sidebar filters (fuel, seller, transmission, year, price)
- ✅ Selling price prediction using Linear Regression (R² = 0.75)
- ✅ Prediction form with depreciation context in the dashboard

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib + Seaborn | Static chart generation |
| Plotly | Interactive charts in dashboard |
| Streamlit | Web dashboard |
| Scikit-learn | Machine learning pipeline |

---

## 📁 Project Structure

```
car_market_trends_analysis/
│
├── data/
│   └── car_dekho_data.csv          ← 301-row cleaned dataset
│
├── models/
│   └── price_predictor.pkl         ← Saved trained model (auto-generated)
│
├── notebooks/                      ← Jupyter exploration space
│
├── src/
│   ├── data_cleaning.py            ← 10-step cleaning pipeline
│   ├── eda.py                      ← 13 EDA analyses + chart generation
│   ├── visualizations.py           ← Static chart exports
│   └── model.py                    ← ML training, evaluation, prediction
│
├── dashboard/
│   └── app.py                      ← 4-page Streamlit dashboard
│
├── report_images/                  ← Auto-generated PNG charts (14 files)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/car-market-trends-analysis.git
cd car-market-trends-analysis/car_market_trends_analysis

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Run data cleaning (inspect the dataset)
```bash
python src/data_cleaning.py
```

### Step 2 — Run EDA (generates 14 charts in report_images/)
```bash
python src/eda.py
```

### Step 3 — Train the prediction model
```bash
python src/model.py
```
> This creates `models/price_predictor.pkl`. Must be done before using the dashboard prediction page.

### Step 4 — Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Then open **http://localhost:8501** in your browser.

---

## 📈 Analysis Results (from actual test data)

### Key EDA Findings

| Insight | Finding |
|---------|---------|
| Most listed model | Honda City (26 listings) |
| Diesel avg price | Rs 10.10L vs Petrol Rs 3.26L |
| Automatic avg price | Rs 9.07L vs Manual Rs 3.92L |
| Dealer avg price | Rs 6.63L vs Individual Rs 0.87L |
| Strongest price predictor | Present_Price (r = 0.876) |
| Dataset year range | 2003 – 2018 |

### Machine Learning Results (on 20% held-out test set)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | Rs 1.47L | Rs 2.52L | **0.75** |
| Random Forest (100 trees) | Rs 1.50L | Rs 3.61L | 0.49 |

**Best model: Linear Regression** with R² = 0.75 (75% of price variance explained).

> ⚠️ These are real metrics from the 60-sample test set. Results may vary slightly
> across runs due to random train/test splitting.

---

## ⚠️ Limitations

- Small dataset: only 299 records — predictions on unusual vehicles may be unreliable
- Mixed vehicle types: dataset includes both cars and 2-wheelers, which affects averages
- No geographic data — prices may vary significantly by city
- CNG has only 2 records — any CNG analysis is statistically insignificant
- Model does not account for car condition, accident history, or colour

---

## 🔮 Future Improvements

- Add more rows (scrape live Car Dekho listings)
- Add model/brand name as a feature (with target encoding)
- Try XGBoost or gradient boosting for better R²
- Add SHAP explainability charts
- Deploy to Streamlit Cloud

---

## 👤 Author

**B.Tech Artificial Intelligence & Data Science Student**

Built as a complete beginner-to-intermediate Python Data Science project covering:
data cleaning, EDA, visualisation, machine learning, and dashboard deployment.

---

*Built with Python · Pandas · Scikit-learn · Plotly · Streamlit*
