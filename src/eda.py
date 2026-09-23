"""
eda.py
======
PURPOSE
-------
Perform a complete Exploratory Data Analysis (EDA) on the cleaned
Car Dekho dataset.  Each analysis is a self-contained function so you
can call them individually or run them all at once.

ANALYSES COVERED
----------------
 1.  Total number of cars
 2.  Number of unique car names
 3.  Distribution of selling prices
 4.  Distribution of present prices
 5.  Car manufacturing year distribution
 6.  Relationship between car age and selling price
 7.  Relationship between kilometres driven and selling price
 8.  Average selling price by fuel type
 9.  Average selling price by seller type
10.  Average selling price by transmission type
11.  Average selling price by number of owners
12.  Top car names by listing frequency
13.  Correlation analysis for numerical columns

All charts are saved as PNG files inside  report_images/

HOW TO RUN
----------
    python src/eda.py          (from the project root folder)
"""

import os
import sys
import warnings

# Use non-interactive Matplotlib backend so charts save without a GUI window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Allow importing data_cleaning from the same src/ folder ───────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import get_clean_data

# ── Where to save the chart images ────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report_images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Consistent visual style for all charts ───────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
TITLE_COLOR  = "#1f2328"
ACCENT_COLOR = "#3b82d4"
GRID_COLOR   = "#e5e7eb"


def _save_chart(fig: plt.Figure, filename: str) -> None:
    """Save a Matplotlib figure to the report_images folder."""
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")


def _section(title: str) -> None:
    """Print a visible section header in the terminal."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 & 2 — Basic counts
# ══════════════════════════════════════════════════════════════════════════════
def basic_counts(df: pd.DataFrame) -> None:
    """
    Print:
      - Total number of car listings in the dataset
      - Number of unique car/bike names
    """
    _section("ANALYSIS 1 & 2 : BASIC COUNTS")

    total_cars   = len(df)
    unique_names = df["Car_Name"].nunique()

    print(f"  Total listings (rows)   : {total_cars}")
    print(f"  Unique car / bike names : {unique_names}")

    # Year range gives context about how old/new the cars are
    print(f"  Manufacture year range  : {df['Year'].min()} – {df['Year'].max()}")
    print(f"  Selling price range     : "
          f"Rs {df['Selling_Price'].min():.2f}L – Rs {df['Selling_Price'].max():.2f}L")

    # Quick categorical split
    print(f"\n  Fuel type breakdown:")
    print(df["Fuel_Type"].value_counts().to_string())
    print(f"\n  Seller type breakdown:")
    print(df["Seller_Type"].value_counts().to_string())
    print(f"\n  Transmission breakdown:")
    print(df["Transmission"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Selling price distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_selling_price_distribution(df: pd.DataFrame) -> None:
    """
    Show how selling prices are spread across the dataset.
    A histogram tells us whether prices cluster at the low end (cheap bikes)
    or are spread evenly.  A KDE curve (smooth line) shows the shape.

    FINDING : Most listings are below Rs 10L because the dataset includes
    many 2-wheelers listed at under Rs 2L.
    """
    _section("ANALYSIS 3 : SELLING PRICE DISTRIBUTION")

    # Summary stats
    print(f"  Mean   : Rs {df['Selling_Price'].mean():.2f}L")
    print(f"  Median : Rs {df['Selling_Price'].median():.2f}L")
    print(f"  Std    : Rs {df['Selling_Price'].std():.2f}L")
    print(f"  Min    : Rs {df['Selling_Price'].min():.2f}L")
    print(f"  Max    : Rs {df['Selling_Price'].max():.2f}L")
    print("  NOTE   : Mean > Median means the distribution is right-skewed "
          "(a few expensive cars pull the average up).")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — full distribution
    sns.histplot(df["Selling_Price"], bins=40, kde=True,
                 color=ACCENT_COLOR, ax=axes[0])
    axes[0].set_title("Selling Price Distribution (All listings)",
                      fontsize=13, color=TITLE_COLOR)
    axes[0].set_xlabel("Selling Price (Rs Lakhs)")
    axes[0].set_ylabel("Number of Listings")

    # Right — zoom in on <= Rs 15L to see the main cluster clearly
    under_15 = df[df["Selling_Price"] <= 15]
    sns.histplot(under_15["Selling_Price"], bins=30, kde=True,
                 color="#7c5cd8", ax=axes[1])
    axes[1].set_title("Selling Price Distribution (Under Rs 15L)",
                      fontsize=13, color=TITLE_COLOR)
    axes[1].set_xlabel("Selling Price (Rs Lakhs)")
    axes[1].set_ylabel("Number of Listings")

    fig.suptitle("Analysis 3 — Selling Price Distribution",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_chart(fig, "eda_03_selling_price_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Present price distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_present_price_distribution(df: pd.DataFrame) -> None:
    """
    Show how the current showroom/market price is distributed.
    Comparing this with the selling price reveals how much value is lost.

    FINDING : Present prices range up to Rs 92.6L (Land Cruiser).
    The vast majority are under Rs 15L.
    """
    _section("ANALYSIS 4 : PRESENT PRICE DISTRIBUTION")

    print(f"  Mean   : Rs {df['Present_Price'].mean():.2f}L")
    print(f"  Median : Rs {df['Present_Price'].median():.2f}L")
    print(f"  Max    : Rs {df['Present_Price'].max():.2f}L")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Present_Price"], bins=40, kde=True,
                 color="#f97316", ax=ax)
    ax.set_title("Present Price Distribution",
                 fontsize=13, color=TITLE_COLOR)
    ax.set_xlabel("Present Price (Rs Lakhs)")
    ax.set_ylabel("Number of Listings")
    plt.tight_layout()
    _save_chart(fig, "eda_04_present_price_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5 — Year distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_year_distribution(df: pd.DataFrame) -> None:
    """
    Count how many cars were manufactured in each year.

    FINDING : The dataset is dominated by cars from 2012–2017, reflecting
    the typical age of second-hand vehicles on the Indian used-car market.
    """
    _section("ANALYSIS 5 : MANUFACTURE YEAR DISTRIBUTION")

    year_counts = df["Year"].value_counts().sort_index()
    print(year_counts.to_string())

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(year_counts.index.astype(str), year_counts.values,
                  color=ACCENT_COLOR, edgecolor="white", linewidth=0.5)

    # Label each bar with its count
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                str(int(height)), ha="center", va="bottom", fontsize=9)

    ax.set_title("Number of Cars by Manufacture Year",
                 fontsize=13, color=TITLE_COLOR)
    ax.set_xlabel("Year of Manufacture")
    ax.set_ylabel("Number of Listings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_chart(fig, "eda_05_year_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6 — Car age vs selling price
# ══════════════════════════════════════════════════════════════════════════════
def plot_age_vs_price(df: pd.DataFrame) -> None:
    """
    Scatter plot: does an older car sell for less?

    FINDING : There is a mild negative trend — newer cars tend to sell
    for more.  But the relationship is not perfectly linear because some
    old luxury vehicles (e.g. Land Cruiser) still command high prices.
    """
    _section("ANALYSIS 6 : CAR AGE vs SELLING PRICE")

    corr = df["Car_Age"].corr(df["Selling_Price"])
    print(f"  Correlation (Car_Age vs Selling_Price) : {corr:.4f}")
    print("  A negative value means older cars tend to have lower selling prices.")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df, x="Car_Age", y="Selling_Price",
                    hue="Fuel_Type", alpha=0.65, s=60,
                    palette="Set1", ax=ax)

    # Add a linear trend line
    z = np.polyfit(df["Car_Age"], df["Selling_Price"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["Car_Age"].min(), df["Car_Age"].max(), 100)
    ax.plot(x_line, p(x_line), color="black",
            linewidth=1.5, linestyle="--", label="Trend line")
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.01, 1), loc="upper left")

    ax.set_title(f"Car Age vs Selling Price  (r = {corr:.2f})",
                 fontsize=13, color=TITLE_COLOR)
    ax.set_xlabel("Car Age (Years since manufacture)")
    ax.set_ylabel("Selling Price (Rs Lakhs)")
    plt.tight_layout()
    _save_chart(fig, "eda_06_age_vs_price.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 7 — Kms driven vs selling price
# ══════════════════════════════════════════════════════════════════════════════
def plot_kms_vs_price(df: pd.DataFrame) -> None:
    """
    Scatter plot: does higher mileage lower the price?

    FINDING : The correlation is very weak (near 0) because the dataset
    mixes cars and 2-wheelers with very different km ranges.  Within
    cars only, higher mileage does slightly lower the price.
    """
    _section("ANALYSIS 7 : KMS DRIVEN vs SELLING PRICE")

    corr = df["Kms_Driven"].corr(df["Selling_Price"])
    print(f"  Correlation (Kms_Driven vs Selling_Price) : {corr:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df, x="Kms_Driven", y="Selling_Price",
                    hue="Fuel_Type", alpha=0.55, s=55,
                    palette="Set2", ax=ax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    ax.set_title(f"Kilometres Driven vs Selling Price  (r = {corr:.2f})",
                 fontsize=13, color=TITLE_COLOR)
    ax.set_xlabel("Kilometres Driven")
    ax.set_ylabel("Selling Price (Rs Lakhs)")
    ax.legend(title="Fuel Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    _save_chart(fig, "eda_07_kms_vs_price.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 8 — Avg selling price by fuel type
# ══════════════════════════════════════════════════════════════════════════════
def plot_price_by_fuel(df: pd.DataFrame) -> None:
    """
    Compare average (mean) and median selling prices across fuel types.

    FINDING : Diesel cars command the highest average price (Rs ~10L) because
    most diesel listings are larger cars (SUVs, sedans).  Petrol-only
    listings include many low-cost 2-wheelers which pull the average down.
    CNG vehicles are too few (2 records) to draw strong conclusions.
    """
    _section("ANALYSIS 8 : AVG SELLING PRICE BY FUEL TYPE")

    summary = (
        df.groupby("Fuel_Type")["Selling_Price"]
          .agg(Count="count", Mean="mean", Median="median", Std="std")
          .round(2)
          .sort_values("Mean", ascending=False)
    )
    print(summary.to_string())
    print("\n  NOTE : CNG has only 2 records — treat with caution.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart — mean price
    order = summary.index.tolist()
    sns.barplot(data=df, x="Fuel_Type", y="Selling_Price",
                order=order, estimator="mean",
                palette="Set2", errorbar="sd", ax=axes[0])
    axes[0].set_title("Mean Selling Price by Fuel Type",
                      fontsize=12, color=TITLE_COLOR)
    axes[0].set_xlabel("Fuel Type")
    axes[0].set_ylabel("Avg Selling Price (Rs Lakhs)")

    # Box plot — distribution shape
    sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price",
                order=order, palette="Set2", ax=axes[1])
    axes[1].set_title("Price Distribution by Fuel Type",
                      fontsize=12, color=TITLE_COLOR)
    axes[1].set_xlabel("Fuel Type")
    axes[1].set_ylabel("Selling Price (Rs Lakhs)")

    fig.suptitle("Analysis 8 — Selling Price by Fuel Type",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_chart(fig, "eda_08_price_by_fuel.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 9 — Avg selling price by seller type
# ══════════════════════════════════════════════════════════════════════════════
def plot_price_by_seller(df: pd.DataFrame) -> None:
    """
    Compare dealer vs individual seller prices.

    FINDING : Dealer listings average around Rs 6.6L while Individual
    listings average around Rs 0.87L.  This is primarily because dealers
    mainly sell cars while individuals mostly sell 2-wheelers in this dataset.
    """
    _section("ANALYSIS 9 : AVG SELLING PRICE BY SELLER TYPE")

    summary = (
        df.groupby("Seller_Type")["Selling_Price"]
          .agg(Count="count", Mean="mean", Median="median")
          .round(2)
    )
    print(summary.to_string())
    print("\n  NOTE : Price gap reflects vehicle type mix, not seller markup alone.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sns.barplot(data=df, x="Seller_Type", y="Selling_Price",
                estimator="mean", palette="pastel",
                errorbar="sd", ax=axes[0])
    axes[0].set_title("Mean Selling Price by Seller Type",
                      fontsize=12, color=TITLE_COLOR)
    axes[0].set_xlabel("Seller Type")
    axes[0].set_ylabel("Avg Selling Price (Rs Lakhs)")

    sns.boxplot(data=df, x="Seller_Type", y="Selling_Price",
                palette="pastel", ax=axes[1])
    axes[1].set_title("Price Distribution by Seller Type",
                      fontsize=12, color=TITLE_COLOR)
    axes[1].set_xlabel("Seller Type")
    axes[1].set_ylabel("Selling Price (Rs Lakhs)")

    fig.suptitle("Analysis 9 — Selling Price by Seller Type",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_chart(fig, "eda_09_price_by_seller.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 10 — Avg selling price by transmission
# ══════════════════════════════════════════════════════════════════════════════
def plot_price_by_transmission(df: pd.DataFrame) -> None:
    """
    Compare manual vs automatic transmission prices.

    FINDING : Automatic cars sell for a significantly higher average price
    (~Rs 9L vs ~Rs 3.9L for manual).  Automatic vehicles tend to be premium
    or imported models, which naturally cost more.
    """
    _section("ANALYSIS 10 : AVG SELLING PRICE BY TRANSMISSION")

    summary = (
        df.groupby("Transmission")["Selling_Price"]
          .agg(Count="count", Mean="mean", Median="median")
          .round(2)
    )
    print(summary.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sns.barplot(data=df, x="Transmission", y="Selling_Price",
                estimator="mean", palette=["#3b82d4", "#7c5cd8"],
                errorbar="sd", ax=axes[0])
    axes[0].set_title("Mean Selling Price by Transmission",
                      fontsize=12, color=TITLE_COLOR)
    axes[0].set_xlabel("Transmission Type")
    axes[0].set_ylabel("Avg Selling Price (Rs Lakhs)")

    sns.violinplot(data=df, x="Transmission", y="Selling_Price",
                   palette=["#3b82d4", "#7c5cd8"],
                   inner="quartile", ax=axes[1])
    axes[1].set_title("Price Distribution by Transmission (Violin)",
                      fontsize=12, color=TITLE_COLOR)
    axes[1].set_xlabel("Transmission Type")
    axes[1].set_ylabel("Selling Price (Rs Lakhs)")

    fig.suptitle("Analysis 10 — Selling Price by Transmission",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_chart(fig, "eda_10_price_by_transmission.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 11 — Avg selling price by owner count
# ══════════════════════════════════════════════════════════════════════════════
def plot_price_by_owner(df: pd.DataFrame) -> None:
    """
    Does the number of previous owners affect the selling price?

    FINDING : First-owner (Owner=0) vehicles sell for the highest average
    price.  The sample for Owner=3 is only 1 record (a Camry), so that
    single data point should not be used to draw conclusions.
    """
    _section("ANALYSIS 11 : AVG SELLING PRICE BY OWNER COUNT")

    summary = (
        df.groupby("Owner")["Selling_Price"]
          .agg(Count="count", Mean="mean", Median="median")
          .round(2)
    )
    print(summary.to_string())
    print("\n  CAUTION : Owner=3 has only 1 record — not statistically meaningful.")

    fig, ax = plt.subplots(figsize=(7, 5))
    avg_by_owner = df.groupby("Owner")["Selling_Price"].mean().reset_index()
    count_map    = df.groupby("Owner").size().to_dict()

    bars = ax.bar(avg_by_owner["Owner"].astype(str),
                  avg_by_owner["Selling_Price"],
                  color=["#3b82d4", "#7c5cd8", "#f97316", "#22c55e"])

    # Annotate each bar with count of records
    for bar, (_, row) in zip(bars, avg_by_owner.iterrows()):
        owner_val = int(row["Owner"])
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"n={count_map[owner_val]}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Average Selling Price by Number of Previous Owners",
                 fontsize=12, color=TITLE_COLOR)
    ax.set_xlabel("Number of Previous Owners")
    ax.set_ylabel("Avg Selling Price (Rs Lakhs)")
    plt.tight_layout()
    _save_chart(fig, "eda_11_price_by_owner.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 12 — Top car names by frequency
# ══════════════════════════════════════════════════════════════════════════════
def plot_top_car_names(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Which car/bike names appear most often in the listings?

    FINDING : Honda City is the most listed car (26 listings), followed by
    Toyota Corolla Altis and Hyundai Verna.  The high frequency of Royal
    Enfield bikes shows that 2-wheelers are well represented.
    """
    _section(f"ANALYSIS 12 : TOP {top_n} CAR NAMES BY LISTING FREQUENCY")

    top = df["Car_Name"].value_counts().head(top_n).sort_values()
    print(top.to_string())

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [ACCENT_COLOR if i >= len(top) - 3 else "#94a3b8"
              for i in range(len(top))]
    top.plot(kind="barh", color=colors, ax=ax)

    # Add count labels
    for i, v in enumerate(top.values):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)

    ax.set_title(f"Top {top_n} Most Listed Cars / Bikes",
                 fontsize=13, color=TITLE_COLOR)
    ax.set_xlabel("Number of Listings")
    ax.set_ylabel("")
    plt.tight_layout()
    _save_chart(fig, "eda_12_top_car_names.png")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 13 — Correlation matrix
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation(df: pd.DataFrame) -> None:
    """
    Calculate and visualise the Pearson correlation between all numeric columns.

    Key correlations with Selling_Price:
      - Present_Price     : Strong positive (r ~ 0.88) — makes sense, more
                            expensive cars retain more value.
      - Year              : Mild positive (r ~ 0.23) — newer cars sell for more.
      - Car_Age           : Mild negative (r ~ -0.23) — older = lower price.
      - Kms_Driven        : Near zero (r ~ 0.03) — weak relationship overall.
      - Owner             : Slight negative (r ~ -0.09) — more owners = cheaper.
    """
    _section("ANALYSIS 13 : CORRELATION ANALYSIS")

    num_cols = [
        "Selling_Price", "Present_Price", "Kms_Driven",
        "Year", "Car_Age", "Owner", "Price_Depreciation"
    ]
    corr = df[num_cols].corr().round(3)

    print("  Correlation matrix:")
    print(corr.to_string())
    print("\n  Correlation with Selling_Price (strongest first):")
    sp_corr = corr["Selling_Price"].drop("Selling_Price").sort_values(
        key=abs, ascending=False
    )
    for feat, val in sp_corr.items():
        direction = "positive" if val > 0 else "negative"
        print(f"    {feat:<22} : {val:+.3f}  ({direction})")

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Heatmap — Numeric Features",
                 fontsize=13, color=TITLE_COLOR, pad=14)
    plt.tight_layout()
    _save_chart(fig, "eda_13_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — Avg price trend by year (line chart)
# ══════════════════════════════════════════════════════════════════════════════
def plot_avg_price_by_year(df: pd.DataFrame) -> None:
    """
    Line chart showing the average selling price for each manufacture year.
    Useful for spotting which vintage years hold their value best.
    """
    _section("BONUS : AVG SELLING PRICE BY MANUFACTURE YEAR")

    yearly = (
        df.groupby("Year")["Selling_Price"]
          .agg(Avg_Price="mean", Count="count")
          .round(2)
          .reset_index()
    )
    print(yearly.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly["Year"], yearly["Avg_Price"],
            marker="o", color=ACCENT_COLOR, linewidth=2.2, markersize=7)
    ax.fill_between(yearly["Year"], yearly["Avg_Price"],
                    alpha=0.12, color=ACCENT_COLOR)

    # Annotate each data point
    for _, row in yearly.iterrows():
        ax.annotate(f"{row['Avg_Price']:.1f}",
                    xy=(row["Year"], row["Avg_Price"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#57606a")

    ax.set_title("Average Selling Price by Manufacture Year",
                 fontsize=13, color=TITLE_COLOR)
    ax.set_xlabel("Year of Manufacture")
    ax.set_ylabel("Avg Selling Price (Rs Lakhs)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_chart(fig, "eda_bonus_avg_price_by_year.png")


# ══════════════════════════════════════════════════════════════════════════════
# Run all analyses
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading and cleaning dataset …")
    df = get_clean_data()
    print(f"Ready. Shape: {df.shape}\n")

    basic_counts(df)
    plot_selling_price_distribution(df)
    plot_present_price_distribution(df)
    plot_year_distribution(df)
    plot_age_vs_price(df)
    plot_kms_vs_price(df)
    plot_price_by_fuel(df)
    plot_price_by_seller(df)
    plot_price_by_transmission(df)
    plot_price_by_owner(df)
    plot_top_car_names(df)
    plot_correlation(df)
    plot_avg_price_by_year(df)

    print("\n" + "=" * 60)
    print("  EDA COMPLETE")
    print(f"  All charts saved to: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)
