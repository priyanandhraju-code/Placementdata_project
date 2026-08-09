import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data
import matplotlib

matplotlib.use("Agg")

sns.set_style("whitegrid")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight", dpi=100)
    plt.close("all")


# CSV_PATH = r"D:\ML\placement_predict_50k Dataset (3)(in).csv"
#
# sns.set_style("whitegrid")
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 200)
#
#
# def show(title=""):
#     if title:
#         plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()
#     plt.show("all")
#
#
# if not os.path.exists(CSV_PATH):
#     raise FileNotFoundError(f"csv not found at '{CSV_PATH}'.Update CSV_PATH at top of script")
# data = pd.read_csv(CSV_PATH)
# print("=" * 80)
# print("1.DATA Loaded")
# print("=" * 80)
# print("Shape:", data.shape)
# print("\nFirst 5-rows:\n", data.head())
#
# print("\n" + "=" * 80)
# print("2.Basic Info")
# print("=" * 80)
# print(data.info())
# print("\nColumn dtypes:\n", data.dtypes)
# print("\nDescribe (numeric):\n", data.describe())
# print("\nDescribe (categorical):\n", data.describe(include="object"))
# print("\n" + "=" * 80)


def run_eda() -> dict:
    data = load_data()
    charts = []
    print("3.Missing values")
    print("=" * 80)
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
    _save("missing_values.png")
    charts.append("missing_values.png")



    if not missing_df.empty:
       plt.figure(figsize=(10, 5), dpi=100)
       sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
       plt.xticks(rotation=45, ha="right")
       plt.ylabel("Missing %")
       plt.title("Missing Values by columns")
       _save("missing_values.png")
       charts.append("missing_values.png")

    print("\n" + "=" * 80)
    print("4.Duplicate Rows")
    print("=" * 80)
    print("Duplicate rows:", data.duplicated().sum())





    print("\n" + "=" * 80)
    print("5.Target Variable - Placement status")
    print("=" * 80)
    print(data["PlacementStatus"].value_counts())
    target_counts = data["PlacementStatus"].value_counts().to_dict()

    plt.figure(figsize=(6, 4), dpi=125)
    sns.countplot(x="PlacementStatus", data=data)
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.title("Placement Status Distribution")

    _save("placement_status.png")
    charts.append("placement_status.png")

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": int(data.duplicated().sum()),
        "missing": missing_df["missing_count"].to_dict(),
        "target_counts": target_counts,
        "charts": charts,
    }



if __name__=="__main__":
    results=run_eda()
    print(results)