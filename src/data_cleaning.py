"""
data_cleaning.py
================
PURPOSE
-------
Load the Car Dekho CSV file, inspect it thoroughly, flag any problems,
and return a clean DataFrame that is safe to use for analysis and
machine learning later.

IMPORTANT DECISIONS
-------------------
* We NEVER delete the original DataFrame.  All cleaning is done on a copy.
* We do NOT impute (fill in) missing values silently.  We report them first.
* We remove only EXACT duplicate rows (all 9 columns identical).
* We add two derived columns (Car_Age, Price_Depreciation) ONLY on the
  cleaned copy so the raw data stays untouched.
* Categorical text is stripped of spaces and title-cased for consistency.

HOW TO RUN
----------
    python src/data_cleaning.py          (from the project root folder)
"""

import os
import pandas as pd
import numpy as np

# ── Locate the CSV file relative to THIS script ────────────────────────────────
# os.path.dirname(__file__)  →  the folder that contains this .py file (src/)
# ".."                       →  one folder up (project root)
# "data"                     →  then into the data/ folder
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "car_dekho_data.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ── Load the CSV
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Read the CSV file and return a raw (unchanged) DataFrame.
    We store this as the 'original' so we can always compare later.
    """
    df = pd.read_csv(path)
    print("=" * 60)
    print("STEP 1 : LOAD DATASET")
    print("=" * 60)
    print(f"  File loaded  : {os.path.abspath(path)}")
    print(f"  Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ── Display shape, column names, and data types
# ══════════════════════════════════════════════════════════════════════════════
def inspect_structure(df: pd.DataFrame) -> None:
    """Print the shape, column list, and data types of the DataFrame."""
    print("\n" + "=" * 60)
    print("STEP 2 : STRUCTURE — COLUMNS & DATA TYPES")
    print("=" * 60)

    print(f"\n  Rows : {df.shape[0]}   Columns : {df.shape[1]}")

    print("\n  Column Name         | Data Type")
    print("  " + "-" * 38)
    for col in df.columns:
        print(f"  {col:<20} | {str(df[col].dtype)}")

    print("\n  First 5 rows:")
    print(df.head().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ── Missing values
# ══════════════════════════════════════════════════════════════════════════════
def check_missing(df: pd.DataFrame) -> None:
    """
    Count missing values (NaN) in every column.
    Missing values can break calculations, so we must find them first.
    """
    print("\n" + "=" * 60)
    print("STEP 3 : MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()          # count of NaN per column
    pct     = (missing / len(df)) * 100  # as a percentage

    report = pd.DataFrame({"Missing Count": missing, "Missing %": pct.round(2)})
    print(report.to_string())

    total_missing = missing.sum()
    if total_missing == 0:
        print("\n  RESULT : No missing values found. Dataset is complete.")
    else:
        print(f"\n  RESULT : {total_missing} missing value(s) found — review before analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ── Duplicate rows
# ══════════════════════════════════════════════════════════════════════════════
def check_duplicates(df: pd.DataFrame) -> None:
    """
    Find rows where ALL column values are identical.
    Duplicates distort statistics (e.g. average price), so we remove them.
    """
    print("\n" + "=" * 60)
    print("STEP 4 : DUPLICATE ROWS")
    print("=" * 60)

    n_dupes = df.duplicated().sum()
    print(f"  Exact duplicate rows found : {n_dupes}")

    if n_dupes > 0:
        print("\n  Duplicate rows (first occurrence shown alongside duplicate):")
        dupe_mask = df.duplicated(keep=False)   # mark ALL copies, not just 2nd
        print(df[dupe_mask].sort_values(list(df.columns)).to_string())
        print(f"\n  DECISION : These {n_dupes} duplicate(s) will be removed.")
    else:
        print("  RESULT : No duplicates found.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 ── Year column
# ══════════════════════════════════════════════════════════════════════════════
def check_year(df: pd.DataFrame) -> None:
    """
    Cars manufactured before 1990 or after the current year are suspicious.
    We flag them but do NOT delete them automatically.
    """
    print("\n" + "=" * 60)
    print("STEP 5 : YEAR COLUMN")
    print("=" * 60)

    print(f"  Min Year : {df['Year'].min()}")
    print(f"  Max Year : {df['Year'].max()}")
    print(f"  Unique values : {sorted(df['Year'].unique())}")

    CURRENT_YEAR = 2024
    invalid_year = df[(df["Year"] < 1990) | (df["Year"] > CURRENT_YEAR)]
    if len(invalid_year) > 0:
        print(f"\n  WARNING : {len(invalid_year)} row(s) with unusual Year values:")
        print(invalid_year[["Car_Name", "Year"]].to_string())
    else:
        print("  RESULT : All Year values look valid (1990 – 2024).")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 ── Selling_Price and Present_Price
# ══════════════════════════════════════════════════════════════════════════════
def check_prices(df: pd.DataFrame) -> None:
    """
    Prices must be positive numbers (in Indian Rupee Lakhs).
    Selling_Price should generally be <= Present_Price (depreciation).
    We report any violations without deleting rows.
    """
    print("\n" + "=" * 60)
    print("STEP 6 : SELLING PRICE & PRESENT PRICE")
    print("=" * 60)

    for col in ["Selling_Price", "Present_Price"]:
        print(f"\n  {col}:")
        print(f"    Min    : {df[col].min():.2f} Lakhs")
        print(f"    Max    : {df[col].max():.2f} Lakhs")
        print(f"    Mean   : {df[col].mean():.2f} Lakhs")
        print(f"    Median : {df[col].median():.2f} Lakhs")

        # Flag zero or negative prices — these are almost certainly errors
        bad = df[df[col] <= 0]
        if len(bad) > 0:
            print(f"    WARNING : {len(bad)} row(s) with price <= 0:")
            print(bad[["Car_Name", col]].to_string())

    # Check: is Selling_Price ever higher than Present_Price?
    # (A used car being sold for MORE than new showroom price is unusual)
    overpriced = df[df["Selling_Price"] > df["Present_Price"]]
    print(f"\n  Rows where Selling_Price > Present_Price : {len(overpriced)}")
    if len(overpriced) > 0:
        print("  (These could be luxury/collector cars or data entry errors)")
        print(overpriced[["Car_Name", "Year", "Selling_Price", "Present_Price"]].to_string())
    else:
        print("  RESULT : All selling prices are at or below present price. Good.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 ── Kms_Driven
# ══════════════════════════════════════════════════════════════════════════════
def check_kms(df: pd.DataFrame) -> None:
    """
    Kilometre values must be positive.
    Extremely high values (e.g. 500,000 km) may be data entry errors.
    We flag them for review.
    """
    print("\n" + "=" * 60)
    print("STEP 7 : KMS DRIVEN")
    print("=" * 60)

    print(f"  Min  : {df['Kms_Driven'].min():,}")
    print(f"  Max  : {df['Kms_Driven'].max():,}")
    print(f"  Mean : {df['Kms_Driven'].mean():,.0f}")

    # Flag anything at or below zero
    bad_kms = df[df["Kms_Driven"] <= 0]
    if len(bad_kms) > 0:
        print(f"\n  WARNING : {len(bad_kms)} row(s) with Kms_Driven <= 0")
        print(bad_kms[["Car_Name", "Kms_Driven"]].to_string())

    # Flag potential outliers using a simple threshold (> 400,000 km)
    HIGH_KMS = 400_000
    high_kms = df[df["Kms_Driven"] > HIGH_KMS]
    if len(high_kms) > 0:
        print(f"\n  NOTE : {len(high_kms)} row(s) with Kms_Driven > {HIGH_KMS:,} (possible outlier):")
        print(high_kms[["Car_Name", "Year", "Kms_Driven"]].to_string())
        print("  DECISION : Kept as-is. Extreme mileage is possible for old vehicles.")
    else:
        print("  RESULT : No extreme Kms_Driven values found.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 ── Categorical columns
# ══════════════════════════════════════════════════════════════════════════════
def check_categoricals(df: pd.DataFrame) -> None:
    """
    Check Fuel_Type, Seller_Type, Transmission, and Owner for:
    - Unexpected values (typos, extra spaces)
    - Value counts (so we know how the data is distributed)
    """
    print("\n" + "=" * 60)
    print("STEP 8 : CATEGORICAL COLUMNS")
    print("=" * 60)

    cat_cols = {
        "Fuel_Type"    : ["Petrol", "Diesel", "CNG"],
        "Seller_Type"  : ["Dealer", "Individual"],
        "Transmission" : ["Manual", "Automatic"],
    }

    for col, valid_values in cat_cols.items():
        print(f"\n  {col}:")
        counts = df[col].str.strip().value_counts()
        print(counts.to_string())

        # Find any values NOT in the expected list
        unique_vals = df[col].str.strip().unique().tolist()
        unexpected = [v for v in unique_vals if v not in valid_values]
        if unexpected:
            print(f"  WARNING : Unexpected values found → {unexpected}")
        else:
            print(f"  OK : All values match expected set {valid_values}")

    # Owner is numeric (0, 1, 2, 3 …) — check its distribution
    print(f"\n  Owner (number of previous owners):")
    print(df["Owner"].value_counts().sort_index().to_string())
    if df["Owner"].min() < 0:
        print("  WARNING : Negative Owner values found — review required.")
    else:
        print("  OK : All Owner values are >= 0.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 ── Apply cleaning and return a clean copy
# ══════════════════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a COPY of the raw DataFrame.
    Returns the cleaned DataFrame.

    What we do:
      a) Strip whitespace from all text columns.
      b) Title-case Car_Name for consistent labelling.
      c) Remove exact duplicate rows.
      d) Add Car_Age  = 2024 - Year
      e) Add Price_Depreciation = Present_Price - Selling_Price
      f) Reset the row index so it starts at 0 again.

    What we do NOT do (to avoid data leakage / silent corruption):
      - We do NOT fill in missing values automatically.
      - We do NOT remove outlier rows without user review.
      - We do NOT modify the original df passed in.
    """
    print("\n" + "=" * 60)
    print("STEP 9 : APPLYING CLEANING STEPS")
    print("=" * 60)

    # Work on a copy — never touch the original
    df_clean = df.copy()

    # a) Strip whitespace from all string/object columns
    str_cols = df_clean.select_dtypes(include="object").columns
    for col in str_cols:
        df_clean[col] = df_clean[col].str.strip()
    print("  [a] Stripped leading/trailing whitespace from text columns.")

    # b) Title-case Car_Name  (e.g. "swift" → "Swift")
    df_clean["Car_Name"] = df_clean["Car_Name"].str.title()
    print("  [b] Standardised Car_Name to title-case.")

    # c) Remove exact duplicate rows
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed = before - len(df_clean)
    print(f"  [c] Removed {removed} exact duplicate row(s). "
          f"({before} → {len(df_clean)} rows)")

    # d) Add Car_Age column
    CURRENT_YEAR = 2024
    df_clean["Car_Age"] = CURRENT_YEAR - df_clean["Year"]
    print(f"  [d] Added 'Car_Age' column (2024 - Year).")

    # e) Add Price_Depreciation column
    df_clean["Price_Depreciation"] = (
        df_clean["Present_Price"] - df_clean["Selling_Price"]
    ).round(2)
    print("  [e] Added 'Price_Depreciation' column (Present - Selling price).")

    # f) Reset index (already done above with drop_duplicates)
    print(f"  [f] Index reset. Final shape: {df_clean.shape[0]} rows x "
          f"{df_clean.shape[1]} columns.")

    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 ── Final summary
# ══════════════════════════════════════════════════════════════════════════════
def cleaning_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """Print a before/after comparison of the raw vs cleaned dataset."""
    print("\n" + "=" * 60)
    print("STEP 10 : CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Original rows      : {len(raw_df)}")
    print(f"  Cleaned rows       : {len(clean_df)}")
    print(f"  Rows removed       : {len(raw_df) - len(clean_df)}")
    print(f"  Original columns   : {raw_df.shape[1]}")
    print(f"  Cleaned columns    : {clean_df.shape[1]}  "
          f"(+{clean_df.shape[1] - raw_df.shape[1]} derived)")
    print(f"\n  Final columns:")
    for col in clean_df.columns:
        tag = " (derived)" if col in ("Car_Age", "Price_Depreciation") else ""
        print(f"    - {col}{tag}")
    print("\n  Dataset is ready for EDA and modelling.")


# ══════════════════════════════════════════════════════════════════════════════
# Public helper — used by eda.py and app.py
# ══════════════════════════════════════════════════════════════════════════════
def get_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    """One-liner helper: load + clean and return the clean DataFrame."""
    return clean_data(load_data(path))


# ══════════════════════════════════════════════════════════════════════════════
# Run all steps when executed directly
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    raw_df   = load_data()
    inspect_structure(raw_df)
    check_missing(raw_df)
    check_duplicates(raw_df)
    check_year(raw_df)
    check_prices(raw_df)
    check_kms(raw_df)
    check_categoricals(raw_df)
    clean_df = clean_data(raw_df)
    cleaning_summary(raw_df, clean_df)
