"""
visualizations.py
-----------------
PURPOSE : Generate and save all static charts (PNG) to the
          report_images/ folder using Matplotlib and Seaborn.

HOW TO RUN (from the project root):
    python src/visualizations.py
"""

import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import get_clean_data

# ── Output folder ──────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report_images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
TITLE_PAD = 14


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾  Saved → {path}")


# ── Chart 1 : Selling Price Distribution ──────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Selling_Price"], bins=30, kde=True, color="#3b82d4", ax=ax)
    ax.set_title("Distribution of Selling Price (₹ Lakhs)", pad=TITLE_PAD)
    ax.set_xlabel("Selling Price (₹ Lakhs)")
    ax.set_ylabel("Count")
    _save(fig, "01_price_distribution.png")


# ── Chart 2 : Selling Price by Fuel Type ──────────────────────────────────────
def plot_price_by_fuel(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df.groupby("Fuel_Type")["Selling_Price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", order=order,
                palette="Set2", ax=ax)
    ax.set_title("Selling Price by Fuel Type", pad=TITLE_PAD)
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Selling Price (₹ Lakhs)")
    _save(fig, "02_price_by_fuel.png")


# ── Chart 3 : Selling Price by Seller Type ────────────────────────────────────
def plot_price_by_seller(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x="Seller_Type", y="Selling_Price",
                palette="pastel", ax=ax)
    ax.set_title("Selling Price by Seller Type", pad=TITLE_PAD)
    ax.set_xlabel("Seller Type")
    ax.set_ylabel("Selling Price (₹ Lakhs)")
    _save(fig, "03_price_by_seller.png")


# ── Chart 4 : Selling Price by Transmission ───────────────────────────────────
def plot_price_by_transmission(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x="Transmission", y="Selling_Price",
                palette="Set1", ax=ax)
    ax.set_title("Selling Price by Transmission Type", pad=TITLE_PAD)
    ax.set_xlabel("Transmission")
    ax.set_ylabel("Selling Price (₹ Lakhs)")
    _save(fig, "04_price_by_transmission.png")


# ── Chart 5 : Car Age vs Selling Price (scatter) ──────────────────────────────
def plot_age_vs_price(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="Car_Age", y="Selling_Price",
                    hue="Fuel_Type", alpha=0.7, ax=ax)
    ax.set_title("Car Age vs Selling Price", pad=TITLE_PAD)
    ax.set_xlabel("Car Age (Years)")
    ax.set_ylabel("Selling Price (₹ Lakhs)")
    _save(fig, "05_age_vs_price.png")


# ── Chart 6 : Kms Driven vs Selling Price (scatter) ───────────────────────────
def plot_kms_vs_price(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="Kms_Driven", y="Selling_Price",
                    hue="Fuel_Type", alpha=0.7, ax=ax)
    ax.set_title("Kilometres Driven vs Selling Price", pad=TITLE_PAD)
    ax.set_xlabel("Kilometres Driven")
    ax.set_ylabel("Selling Price (₹ Lakhs)")
    _save(fig, "06_kms_vs_price.png")


# ── Chart 7 : Fuel Type Share (pie chart) ─────────────────────────────────────
def plot_fuel_share(df: pd.DataFrame) -> None:
    counts = df["Fuel_Type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           startangle=140, colors=sns.color_palette("Set2", len(counts)))
    ax.set_title("Fuel Type Share in Used-Car Market", pad=TITLE_PAD)
    _save(fig, "07_fuel_share.png")


# ── Chart 8 : Avg Selling Price by Year ───────────────────────────────────────
def plot_avg_price_by_year(df: pd.DataFrame) -> None:
    yearly = df.groupby("Year")["Selling_Price"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=yearly, x="Year", y="Selling_Price",
                 marker="o", color="#3b82d4", ax=ax)
    ax.set_title("Average Selling Price by Manufacture Year", pad=TITLE_PAD)
    ax.set_xlabel("Year of Manufacture")
    ax.set_ylabel("Avg Selling Price (₹ Lakhs)")
    plt.xticks(rotation=45)
    _save(fig, "08_avg_price_by_year.png")


# ── Chart 9 : Top 10 Car Models ───────────────────────────────────────────────
def plot_top_models(df: pd.DataFrame) -> None:
    top10 = df["Car_Name"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    top10.plot(kind="barh", color="#7c5cd8", ax=ax)
    ax.set_title("Top 10 Most Listed Car/Bike Models", pad=TITLE_PAD)
    ax.set_xlabel("Number of Listings")
    ax.set_ylabel("Car / Bike Name")
    _save(fig, "09_top_models.png")


# ── Chart 10 : Correlation Heatmap ────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = ["Selling_Price", "Present_Price", "Kms_Driven",
                "Car_Age", "Year", "Owner", "Price_Depreciation"]
    corr = df[num_cols].corr().round(2)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", pad=TITLE_PAD)
    _save(fig, "10_correlation_heatmap.png")


# ── Run all charts ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("⏳  Generating all charts …\n")
    df = get_clean_data()

    plot_price_distribution(df)
    plot_price_by_fuel(df)
    plot_price_by_seller(df)
    plot_price_by_transmission(df)
    plot_age_vs_price(df)
    plot_kms_vs_price(df)
    plot_fuel_share(df)
    plot_avg_price_by_year(df)
    plot_top_models(df)
    plot_correlation_heatmap(df)

    print(f"\n✅  All 10 charts saved to → {os.path.abspath(OUT_DIR)}")
