"""
model.py
========
PURPOSE
-------
Train a used-car selling price prediction model on the cleaned Car Dekho
dataset and evaluate it honestly on held-out test data.

WHAT THIS FILE DOES — STEP BY STEP
------------------------------------
 1. Load & clean data  (calls data_cleaning.py)
 2. Feature engineering  (encode categoricals, select columns)
 3. Train / test split  (no data leakage)
 4. Train two models    (Linear Regression  +  Random Forest Regressor)
 5. Evaluate both       (MAE, RMSE, R²)
 6. Save the best model to  models/price_predictor.pkl
 7. Expose a predict()  helper used by the Streamlit dashboard

IMPORTANT DECISIONS
--------------------
* We use ONE-HOT ENCODING for categorical columns (Fuel_Type, Seller_Type,
  Transmission). This converts categories to 0/1 numbers that models can
  understand, without implying any ordering between categories.
* The encoding is fitted ONLY on training data, then applied to test data.
  This avoids data leakage — the model never sees test-set information
  during training.
* Owner and Car_Age are already numeric, so no encoding is needed for them.
* We DO NOT use Car_Name — there are too many unique values and the model
  would overfit to names it has seen.

HOW TO RUN
----------
    python src/model.py          (from the project root folder)
"""

import os
import sys
import warnings
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer

warnings.filterwarnings("ignore")

# Allow importing data_cleaning from the same src/ folder
sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import get_clean_data

# Where to save the trained model
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "price_predictor.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature selection
# ══════════════════════════════════════════════════════════════════════════════
# These are the input features the model will learn from.
# We chose them because they are naturally available when a seller lists a car.
#
# NUMERIC features (numbers the model can use directly):
NUMERIC_FEATURES = [
    "Present_Price",   # Current showroom price — strongest predictor (r=0.88)
    "Kms_Driven",      # Total kilometres driven
    "Owner",           # Number of previous owners
    "Car_Age",         # Age in years (derived from Year)
]

# CATEGORICAL features (text labels that need to be encoded into numbers):
CATEGORICAL_FEATURES = [
    "Fuel_Type",       # Petrol / Diesel / CNG
    "Seller_Type",     # Dealer / Individual
    "Transmission",    # Manual / Automatic
]

# Target variable — what we want to predict
TARGET = "Selling_Price"

# All features combined
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Prepare features and target
# ══════════════════════════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame):
    """
    Select the feature columns (X) and the target column (y).
    Returns X (DataFrame) and y (Series).
    No encoding is done here — that happens inside the pipeline.
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build a preprocessing + model pipeline
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(model):
    """
    Build a scikit-learn Pipeline that:
      a) One-hot encodes the categorical columns
      b) Passes numeric columns through unchanged
      c) Feeds the result into the given regression model

    Using a Pipeline ensures:
      - The encoder is always fitted on training data only.
      - The same transformation is applied consistently to any new data.
      - There is no risk of data leakage.
    """
    # One-hot encoder: converts each category value into a separate 0/1 column
    # handle_unknown='ignore' means unseen categories in future data won't crash
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # ColumnTransformer applies different transformations to different columns
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough",   NUMERIC_FEATURES),     # keep numeric as-is
        ("cat", cat_transformer, CATEGORICAL_FEATURES),  # encode categoricals
    ])

    # Full pipeline: preprocess → model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    model),
    ])
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluation metrics explained
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(name: str, y_true, y_pred) -> dict:
    """
    Calculate and print three standard regression metrics.

    MAE  (Mean Absolute Error)
         Average of |predicted - actual|.
         Easy to interpret: "On average, predictions are off by Rs X Lakhs."
         Lower is better.

    RMSE (Root Mean Squared Error)
         Square root of the average of (predicted - actual)².
         Penalises large errors more than MAE does.
         Lower is better.

    R²   (R-squared / Coefficient of Determination)
         Proportion of variance in the target explained by the model.
         Range: 0.0 (no better than mean prediction) to 1.0 (perfect).
         Higher is better.  Above 0.85 is generally considered good for price.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  {name}")
    print(f"    MAE  (Mean Absolute Error)   : Rs {mae:.4f} Lakhs")
    print(f"    RMSE (Root Mean Sq. Error)   : Rs {rmse:.4f} Lakhs")
    print(f"    R²   (R-squared)             : {r2:.4f}  ({r2*100:.1f}% variance explained)")

    # Interpret R² for the user
    if r2 >= 0.90:
        interp = "Excellent fit"
    elif r2 >= 0.75:
        interp = "Good fit"
    elif r2 >= 0.50:
        interp = "Moderate fit — useful but room to improve"
    else:
        interp = "Weak fit — model needs improvement"
    print(f"    Interpretation               : {interp}")

    return {"name": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train, evaluate, and save
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate() -> dict:
    """
    Full ML workflow: load data → split → train both models → evaluate → save best.
    Returns a dict with evaluation results and the trained pipeline.
    """

    # ── Load clean data ───────────────────────────────────────────────────────
    _section("STEP 1 : LOAD DATA")
    df = get_clean_data()
    print(f"  Clean dataset shape : {df.shape}")

    # ── Feature / target split ────────────────────────────────────────────────
    _section("STEP 2 : FEATURE SELECTION")
    X, y = prepare_features(df)
    print(f"  Features used ({len(ALL_FEATURES)}) : {ALL_FEATURES}")
    print(f"  Target              : {TARGET}")
    print(f"  X shape : {X.shape}    y shape : {y.shape}")

    # ── Train / test split ────────────────────────────────────────────────────
    # test_size=0.20 means 20 % of rows are held out for testing.
    # random_state=42 makes the split reproducible (same split every run).
    # We split BEFORE any fitting — no test data touches the encoder.
    _section("STEP 3 : TRAIN / TEST SPLIT  (80% train, 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print("  NOTE : Encoder will be fitted on training data only.")

    # ── Train and evaluate both models ────────────────────────────────────────
    _section("STEP 4 : TRAIN MODELS")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest (100 trees)": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,            # use all CPU cores
        ),
    }

    results = []
    trained_pipelines = {}

    for model_name, model_obj in models.items():
        print(f"\n  Training: {model_name} …")
        pipeline = build_pipeline(model_obj)
        pipeline.fit(X_train, y_train)           # learn from training data only
        y_pred = pipeline.predict(X_test)        # predict on unseen test data
        metrics = evaluate_model(model_name, y_test, y_pred)
        results.append(metrics)
        trained_pipelines[model_name] = pipeline

    # ── Pick best model (highest R²) ─────────────────────────────────────────
    _section("STEP 5 : SELECT BEST MODEL")
    best = max(results, key=lambda r: r["R2"])
    best_pipeline = trained_pipelines[best["name"]]
    print(f"  Best model : {best['name']}")
    print(f"    R² = {best['R2']}   MAE = Rs {best['MAE']}L   RMSE = Rs {best['RMSE']}L")

    # ── Feature importance (Random Forest only) ───────────────────────────────
    if "Random Forest" in best["name"]:
        _section("STEP 5b : FEATURE IMPORTANCES (Random Forest)")
        rf_model   = best_pipeline.named_steps["regressor"]
        preprocessor = best_pipeline.named_steps["preprocessor"]

        # Get one-hot encoded feature names
        cat_names = (
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
            .tolist()
        )
        all_names = NUMERIC_FEATURES + cat_names
        importances = rf_model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": all_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        print(imp_df.to_string(index=False))

    # ── Save best model ───────────────────────────────────────────────────────
    _section("STEP 6 : SAVE MODEL")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"  Saved to : {os.path.abspath(MODEL_PATH)}")

    return {
        "results": results,
        "best_name": best["name"],
        "best_pipeline": best_pipeline,
        "feature_names": ALL_FEATURES,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Predict helper (used by Streamlit dashboard)
# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    """
    Load the saved model from disk.
    Call this in the dashboard to avoid retraining on every page load.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run 'python src/model.py' first to train and save the model."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_price(
    pipeline,
    present_price: float,
    kms_driven: int,
    owner: int,
    car_age: int,
    fuel_type: str,
    seller_type: str,
    transmission: str,
) -> float:
    """
    Predict the selling price for a single car.

    Parameters
    ----------
    pipeline      : the trained scikit-learn pipeline (from load_model())
    present_price : current showroom price in Rs Lakhs
    kms_driven    : total kilometres driven
    owner         : number of previous owners (0, 1, 2 …)
    car_age       : age of the car in years
    fuel_type     : 'Petrol', 'Diesel', or 'CNG'
    seller_type   : 'Dealer' or 'Individual'
    transmission  : 'Manual' or 'Automatic'

    Returns
    -------
    Predicted selling price in Rs Lakhs (float, rounded to 2 dp).
    """
    row = pd.DataFrame([{
        "Present_Price" : present_price,
        "Kms_Driven"    : kms_driven,
        "Owner"         : owner,
        "Car_Age"       : car_age,
        "Fuel_Type"     : fuel_type,
        "Seller_Type"   : seller_type,
        "Transmission"  : transmission,
    }])
    prediction = pipeline.predict(row)[0]
    # Clamp to a sensible minimum — prices can't be negative
    return round(max(0.0, prediction), 2)


# ══════════════════════════════════════════════════════════════════════════════
# Run as a standalone script
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    output = train_and_evaluate()

    # Demonstrate the predict helper
    _section("DEMO : SINGLE PREDICTION")
    pipeline = output["best_pipeline"]
    demo_price = predict_price(
        pipeline,
        present_price=9.85,
        kms_driven=15000,
        owner=0,
        car_age=7,
        fuel_type="Petrol",
        seller_type="Dealer",
        transmission="Manual",
    )
    print(f"  Demo input  : Present=9.85L, Kms=15000, Owner=0, Age=7yr, Petrol, Dealer, Manual")
    print(f"  Predicted selling price : Rs {demo_price} Lakhs")

    print("\n" + "=" * 60)
    print("  MODEL TRAINING COMPLETE")
    print("=" * 60)
