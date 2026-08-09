import os

import args
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
CSV_PATH = r"placement_predict_50k Dataset (3)(in).csv"
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
def show(title=""):
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close("all")
    #1.load data
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("csv file not found")
data = pd.read_csv(CSV_PATH)
# print("=" *80)
# print("1. Data Loaded")
# print("=" *80)
# print("shape of data: ", data.shape)
#
# print("\nfirst 5 rows :\n",data.head())
#
# #2 BASIC INFO
# print("\n" + "=" * 80)
# print("2. BASIC INFO")
# print("=" * 80)
# print(data.info())
# print("\nColumn dtypes:\n", data.dtypes)
# print("\nDescribe (numeric): \n", data.describe())
# print("\nDescribe (categorical): \n", data.describe(include="object"))
#
# #MISSING VALUES
# print("\n" + "=" * 80)
# print("3.MISSING VALUES")
# print("=" * 80)
# missing = data.isnull().sum()
# missing_pct=(missing/len(data))*100
# missing_df = pd.DataFrame({ "missing_count": missing,"missing pct": missing_pct })
# missing_df=missing_df[missing_df["missing_count"]>0].sort_values(by="missing pct",ascending=False)
# print(missing_df)
#
# if not missing_df.empty:
#     plt.figure(figsize = (10,5), dpi=100)
#     sns.barplot(x=missing_df.index,y=missing_df["missing_count"])
#     plt.xticks(rotation=45,ha="right")
#     plt.ylabel("Missing %")
#     plt.title("Missing value by columns")
#     show()
#
# # 4 DUPILATE
# print("\n" + "=" * 80)
# print("4. DUPLICATE ROWS")
# print("=" * 80)
# print("Duplicate rows: \n", data.duplicated().sum())
#
# # 5. TAEGET VARIABLE DISTRIBUTION (PLACEMENT STATUS)
# print("\n" + "=" * 80)
# print("5.TARGET VARIABLE -PlacementStatus")
# print("=" * 80)
# print(data["PlacementStatus"].value_counts())
#
# plt.figure(dpi=125)
# sns.countplot(x="PlacementStatus", data=data)
# plt.xlabel("Placement Status(0= Not Placed ,1=placed)")
# plt.ylabel("Count")
# plt.title("Count of Placement Status")
# show()
# 6. NUMERIC FEATURES DISTRIBUTION
# print("\n" + "=" * 80)
# print("6. NUMERIC DISTRIBUTION")
# print("=" * 80)
# hist_cols=["CGPA","AttendancePercent","AptitudeTestScore","SoftSkillRating","CodingTestScore","MockInterview"]
# hist_cols=[c for c in hist_cols if c in data.columns]
# data[hist_cols].hist(figsize=(14,10),bins=20)
# show("Numeric Feature Distributions")
#
# #Mean Line Example
# plt.figure(dpi = 125)
# sns.histplot(data["CGPA"], kde = True)
# plt.axvline(x = np.mean(data["CGPA"]), color = "green", linestyle = "--", label = "CGPA" )
# plt.legend()
# plt.title("CGPA DISTRIBUTION")
# show()

# #7. Outlier Detection (BOXPLOTS)
# print("\n" + "=" * 80)
# print("7. OUTLIER DETECTION (BOXPLOTS)")
# print("=" * 80)
#
# box_cols=["CGPA","AttendancePercent","AptitudeTestScore","SoftSkillRating","CodingTestScore","MockInterview","Salary Package"]
# box_cols=[c for c in box_cols if c in data.columns]
#
# for col in box_cols:
#     plt.figure(figsize = (10, 4))
#     sns.boxplot( x = data[col], color = "red")
#     plt.title(f"Box Plot for {col}", fontsize = 20)
#     show()

#8. CORRELATION HEATMAP
# print("\n" + "=" * 80)
# print("8. CORRELATION ANALYSIS")
# print("=" * 80)
#
# corr = data.select_dtypes(include=[np.number]).corr()
# print(np.round(corr, decimals=2 ))
#
# plt.figure(figsize = (16, 12), dpi=100)
# sns.heatmap(np.round(corr, decimals=2), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap")
# show()

# print("\n" + "=" * 80)
# print("9. RELATIONSHIP PLOTS")
# print("=" * 80)
#
# # CGPA vs Salary Package
# if "CGPA" in data.columns and "Salary Package" in data.columns:
#     plt.figure(figsize=(7,5))
#     sns.regplot(x="CGPA", y="Salary Package", data=data,
#                 scatter_kws={"alpha":0.5}, line_kws={"color":"red"})
#     plt.title("CGPA vs Salary Package")
#     show()
#
# # Aptitude vs Coding Test Score
# if "AptitudeTestScore" in data.columns and "CodingTestScore" in data.columns:
#     plt.figure(figsize=(7,5))
#     sns.regplot(x="AptitudeTestScore", y="CodingTestScore", data=data,
#                 scatter_kws={"alpha":0.5}, line_kws={"color":"blue"})
#     plt.title("Aptitude Test Score vs Coding Test Score")
#     show()

# print("\n" + "=" * 80)
# print("10. CATEGORICAL FEATURE COUNTS")
# print("=" * 80)
#
# cat_cols = [
#     "Gender",
#     "City",
#     "CollegeTier",
#     "Stream",
#     "Specialisation",
#     "Hostel",
#     "HistoryOfBacklogs",
#     "CGPA_Tier"
# ]
#
# cat_cols = [c for c in cat_cols if c in data.columns]
#
# for col in cat_cols:
#     plt.figure(figsize=(8,5))
#     sns.countplot(data=data, x=col, order=data[col].value_counts().index)
#     plt.xticks(rotation=45)
#     plt.title(f"{col} Distribution")
#     show()

# print("\n" + "=" * 80)
# print("11. GENDER VS PLACEMENT STATUS")
# print("=" * 80)
#
# if "Gender" in data.columns and "PlacementStatus" in data.columns:
#     plt.figure(figsize=(7,5))
#     sns.countplot(data=data,
#                   x="Gender",
#                   hue="PlacementStatus")
#     plt.title("Gender vs Placement Status")
#     show()

# print("\n" + "=" * 80)
# print("12. COLLEGE TIER / STREAM VS PLACEMENT")
# print("=" * 80)
#
# if "CollegeTier" in data.columns:
#     plt.figure(figsize=(7,5))
#     sns.countplot(data=data,
#                   x="CollegeTier",
#                   hue="PlacementStatus")
#     plt.title("College Tier vs Placement Status")
#     show()
#
# if "Stream" in data.columns:
#     plt.figure(figsize=(10,5))
#     sns.countplot(data=data,
#                   x="Stream",
#                   hue="PlacementStatus")
#     plt.xticks(rotation=45)
#     plt.title("Stream vs Placement Status")
#     show()

# print("\n" + "=" * 80)
# print("13. SGPA TREND")
# print("=" * 80)
#
# sgpa_cols = [
#     "Sem1_SGPA",
#     "Sem2_SGPA",
#     "Sem3_SGPA",
#     "Sem4_SGPA",
#     "Sem5_SGPA",
#     "Sem6_SGPA",
#     "Sem7_SGPA",
#     "Sem8_SGPA"
# ]
#
# sgpa_cols = [c for c in sgpa_cols if c in data.columns]
#
# if len(sgpa_cols) > 0:
#     avg_sgpa = data[sgpa_cols].mean()
#
#     plt.figure(figsize=(9,5))
#     plt.plot(avg_sgpa.index,
#              avg_sgpa.values,
#              marker="o")
#     plt.title("Average SGPA Across Semesters")
#     plt.xlabel("Semester")
#     plt.ylabel("Average SGPA")
#     plt.grid(True)
#     show()


# print("\n" + "=" * 80)
# print("14. SALARY PACKAGE ANALYSIS")
# print("=" * 80)
#
# if "Salary Package" in data.columns:
#
#     placed = data[data["PlacementStatus"] == 1]
#
#     plt.figure(figsize=(8,5))
#     sns.histplot(placed["Salary Package"],
#                  bins=20,
#                  kde=True)
#     plt.title("Salary Distribution (Placed Students)")
#     show()
#
#     if "CollegeTier" in data.columns:
#         plt.figure(figsize=(8,5))
#         sns.boxplot(x="CollegeTier",
#                     y="Salary Package",
#                     data=placed)
#         plt.title("Salary Package by College Tier")
#         show()


print("\n" + "=" * 80)
print("15. PAIRPLOT")
print("=" * 80)

pair_cols = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterview",
    "PlacementStatus"
]

pair_cols = [c for c in pair_cols if c in data.columns]

sns.pairplot(
    data[pair_cols],
    hue="PlacementStatus",
    diag_kind="hist"
)

plt.show()