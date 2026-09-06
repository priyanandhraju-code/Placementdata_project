import os

import numpy as np
import pandas as pd

import matplotlib

# ============================================================
# IMPORTANT FOR FLASK + MACOS
# ============================================================
# Flask runs request handling outside the Matplotlib GUI
# main thread. Agg is a non-GUI backend used to generate
# images without opening macOS windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = r"placement_predict_50k Dataset (3)(in).csv"

# Folder where EDA graphs will be saved
GRAPH_DIR = os.path.join("static", "eda")

os.makedirs(GRAPH_DIR, exist_ok=True)

sns.set(style="whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_plot(filename):
    """
    Save the current matplotlib figure inside static/eda/
    """

    path = os.path.join(GRAPH_DIR, filename)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()

    return "/" + path.replace("\\", "/")


# ============================================================
# MAIN EDA FUNCTION
# ============================================================

def run_eda():

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    if not os.path.exists(CSV_PATH):

        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)

    results = {}


    # ========================================================
    # 2. BASIC DATA INFORMATION
    # ========================================================

    results["rows"] = int(
        data.shape[0]
    )

    results["columns_count"] = int(
        data.shape[1]
    )

    results["columns"] = list(
        data.columns
    )

    results["data_types"] = {
        column: str(dtype)
        for column, dtype in data.dtypes.items()
    }


    # ========================================================
    # 3. MISSING VALUES
    # ========================================================

    missing = data.isnull().sum()

    missing_pct = (
        missing / len(data)
    ) * 100

    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ]

    missing_df = missing_df.sort_values(
        by="missing_pct",
        ascending=False
    )

    results["missing_values"] = (
        missing_df.to_dict("records")
    )


    # --------------------------------------------------------
    # Missing value graph
    # --------------------------------------------------------

    if not missing_df.empty:

        plt.figure(
            figsize=(10, 5)
        )

        sns.barplot(
            data=missing_df,
            x="column",
            y="missing_count"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.xlabel(
            "Columns"
        )

        plt.ylabel(
            "Missing Count"
        )

        plt.title(
            "Missing Values by Column"
        )

        results["missing_plot"] = save_plot(
            "missing_values.png"
        )

    else:

        results["missing_plot"] = None


    # ========================================================
    # 4. DUPLICATE ROWS
    # ========================================================

    duplicate_count = int(
        data.duplicated().sum()
    )

    results["duplicates"] = (
        duplicate_count
    )


    # ========================================================
    # 5. TARGET VARIABLE
    # ========================================================

    if "PlacementStatus" in data.columns:

        placement_counts = (
            data["PlacementStatus"]
            .value_counts()
            .sort_index()
        )

        results["placement_status"] = {
            str(key): int(value)
            for key, value in placement_counts.items()
        }


        # ----------------------------------------------------
        # PlacementStatus graph
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        sns.countplot(
            data=data,
            x="PlacementStatus"
        )

        plt.xlabel(
            "Placement Status "
            "(0 = Not Placed, 1 = Placed)"
        )

        plt.ylabel(
            "Number of Students"
        )

        plt.title(
            "Placement Status Distribution"
        )

        results["placement_plot"] = save_plot(
            "placement_status.png"
        )

    else:

        results["placement_status"] = {}

        results["placement_plot"] = None


    # ========================================================
    # 6. NUMERIC FEATURES
    # ========================================================

    numeric_columns = list(
        data.select_dtypes(
            include=[np.number]
        ).columns
    )

    results["numeric_columns"] = (
        numeric_columns
    )


    # ========================================================
    # 7. DESCRIPTIVE STATISTICS
    # ========================================================

    if numeric_columns:

        statistics = (
            data[numeric_columns]
            .describe()
            .round(2)
        )

        results["statistics"] = (
            statistics.to_dict()
        )

    else:

        results["statistics"] = {}


    # ========================================================
    # 8. NUMERIC FEATURE DISTRIBUTIONS
    # ========================================================

    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillRating",
        "CodingTestScore",
        "MockInterview"
    ]

    hist_cols = [
        column
        for column in hist_cols
        if column in data.columns
    ]

    if hist_cols:

        data[hist_cols].hist(
            figsize=(14, 10),
            bins=20
        )

        plt.suptitle(
            "Numeric Feature Distributions"
        )

        results["distribution_plot"] = save_plot(
            "numeric_distributions.png"
        )

    else:

        results["distribution_plot"] = None


    # ========================================================
    # 9. CGPA DISTRIBUTION
    # ========================================================

    if "CGPA" in data.columns:

        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(
            data["CGPA"],
            kde=True
        )

        plt.axvline(
            x=data["CGPA"].mean(),
            linestyle="--",
            label="Mean CGPA"
        )

        plt.legend()

        plt.xlabel(
            "CGPA"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.title(
            "CGPA Distribution"
        )

        results["cgpa_plot"] = save_plot(
            "cgpa_distribution.png"
        )

    else:

        results["cgpa_plot"] = None


    # ========================================================
    # 10. OUTLIER DETECTION
    # ========================================================

    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillRating",
        "CodingTestScore",
        "MockInterview",
        "Salary Package"
    ]

    box_cols = [
        column
        for column in box_cols
        if column in data.columns
    ]

    results["boxplots"] = []


    for index, column in enumerate(box_cols):

        plt.figure(
            figsize=(10, 4)
        )

        sns.boxplot(
            x=data[column]
        )

        plt.title(
            f"Box Plot for {column}"
        )

        filename = (
            f"boxplot_{index}.png"
        )

        plot_path = save_plot(
            filename
        )

        results["boxplots"].append({
            "column": column,
            "image": plot_path
        })


    # ========================================================
    # 11. CORRELATION ANALYSIS
    # ========================================================

    if len(numeric_columns) > 1:

        corr = (
            data[numeric_columns]
            .corr()
            .round(2)
        )

        results["correlation"] = (
            corr.to_dict()
        )


        plt.figure(
            figsize=(16, 12)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(
            "Correlation Heatmap"
        )

        results["correlation_plot"] = save_plot(
            "correlation_heatmap.png"
        )

    else:

        results["correlation"] = {}

        results["correlation_plot"] = None


    # ========================================================
    # 12. RELATIONSHIP PLOTS
    # ========================================================

    results["relationship_plots"] = []


    # --------------------------------------------------------
    # CGPA vs Salary Package
    # --------------------------------------------------------

    if (
        "CGPA" in data.columns
        and "Salary Package" in data.columns
    ):

        plt.figure(
            figsize=(7, 5)
        )

        sns.regplot(
            x="CGPA",
            y="Salary Package",
            data=data,
            scatter_kws={"alpha": 0.5}
        )

        plt.title(
            "CGPA vs Salary Package"
        )

        path = save_plot(
            "cgpa_vs_salary.png"
        )

        results["relationship_plots"].append({
            "title": "CGPA vs Salary Package",
            "image": path
        })


    # --------------------------------------------------------
    # Aptitude vs Coding
    # --------------------------------------------------------

    if (
        "AptitudeTestScore" in data.columns
        and "CodingTestScore" in data.columns
    ):

        plt.figure(
            figsize=(7, 5)
        )

        sns.regplot(
            x="AptitudeTestScore",
            y="CodingTestScore",
            data=data,
            scatter_kws={"alpha": 0.5}
        )

        plt.title(
            "Aptitude Test Score vs Coding Test Score"
        )

        path = save_plot(
            "aptitude_vs_coding.png"
        )

        results["relationship_plots"].append({
            "title": "Aptitude vs Coding Test Score",
            "image": path
        })


    # ========================================================
    # 13. CATEGORICAL FEATURE COUNTS
    # ========================================================

    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    cat_cols = [
        column
        for column in cat_cols
        if column in data.columns
    ]

    results["categorical_plots"] = []


    for index, column in enumerate(cat_cols):

        plt.figure(
            figsize=(8, 5)
        )

        order = (
            data[column]
            .value_counts()
            .index
        )

        sns.countplot(
            data=data,
            x=column,
            order=order
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.title(
            f"{column} Distribution"
        )

        filename = (
            f"categorical_{index}.png"
        )

        path = save_plot(
            filename
        )

        results["categorical_plots"].append({
            "column": column,
            "image": path
        })


    # ========================================================
    # 14. GENDER VS PLACEMENT
    # ========================================================

    if (
        "Gender" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(
            figsize=(7, 5)
        )

        sns.countplot(
            data=data,
            x="Gender",
            hue="PlacementStatus"
        )

        plt.title(
            "Gender vs Placement Status"
        )

        results["gender_placement_plot"] = save_plot(
            "gender_vs_placement.png"
        )

    else:

        results["gender_placement_plot"] = None


    # ========================================================
    # 15. COLLEGE TIER VS PLACEMENT
    # ========================================================

    if (
        "CollegeTier" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(
            figsize=(7, 5)
        )

        sns.countplot(
            data=data,
            x="CollegeTier",
            hue="PlacementStatus"
        )

        plt.title(
            "College Tier vs Placement Status"
        )

        results["college_tier_plot"] = save_plot(
            "college_tier_vs_placement.png"
        )

    else:

        results["college_tier_plot"] = None


    # ========================================================
    # 16. STREAM VS PLACEMENT
    # ========================================================

    if (
        "Stream" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(
            figsize=(10, 5)
        )

        sns.countplot(
            data=data,
            x="Stream",
            hue="PlacementStatus"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.title(
            "Stream vs Placement Status"
        )

        results["stream_placement_plot"] = save_plot(
            "stream_vs_placement.png"
        )

    else:

        results["stream_placement_plot"] = None


    # ========================================================
    # 17. SGPA TREND
    # ========================================================

    sgpa_cols = [
        "Sem1_SGPA",
        "Sem2_SGPA",
        "Sem3_SGPA",
        "Sem4_SGPA",
        "Sem5_SGPA",
        "Sem6_SGPA",
        "Sem7_SGPA",
        "Sem8_SGPA"
    ]

    sgpa_cols = [
        column
        for column in sgpa_cols
        if column in data.columns
    ]


    if sgpa_cols:

        avg_sgpa = data[
            sgpa_cols
        ].mean()

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            avg_sgpa.index,
            avg_sgpa.values,
            marker="o"
        )

        plt.title(
            "Average SGPA Across Semesters"
        )

        plt.xlabel(
            "Semester"
        )

        plt.ylabel(
            "Average SGPA"
        )

        plt.xticks(
            rotation=45
        )

        plt.grid(
            True
        )

        results["sgpa_plot"] = save_plot(
            "sgpa_trend.png"
        )

    else:

        results["sgpa_plot"] = None


    # ========================================================
    # 18. SALARY PACKAGE ANALYSIS
    # ========================================================

    results["salary_plot"] = None

    results["salary_college_plot"] = None


    if (
        "Salary Package" in data.columns
        and "PlacementStatus" in data.columns
    ):

        placed = data[
            data["PlacementStatus"] == 1
        ]


        if not placed.empty:

            # ------------------------------------------------
            # Salary distribution
            # ------------------------------------------------

            plt.figure(
                figsize=(8, 5)
            )

            sns.histplot(
                placed["Salary Package"],
                bins=20,
                kde=True
            )

            plt.title(
                "Salary Distribution "
                "(Placed Students)"
            )

            plt.xlabel(
                "Salary Package"
            )

            plt.ylabel(
                "Number of Students"
            )

            results["salary_plot"] = save_plot(
                "salary_distribution.png"
            )


            # ------------------------------------------------
            # Salary by college tier
            # ------------------------------------------------

            if "CollegeTier" in data.columns:

                plt.figure(
                    figsize=(8, 5)
                )

                sns.boxplot(
                    x="CollegeTier",
                    y="Salary Package",
                    data=placed
                )

                plt.title(
                    "Salary Package by College Tier"
                )

                results["salary_college_plot"] = save_plot(
                    "salary_by_college_tier.png"
                )


    # ========================================================
    # 19. PAIRPLOT
    # ========================================================

    pair_cols = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterview",
        "PlacementStatus"
    ]

    pair_cols = [
        column
        for column in pair_cols
        if column in data.columns
    ]

    results["pairplot"] = None


    if (
        len(pair_cols) >= 2
        and "PlacementStatus" in pair_cols
    ):

        pairplot = sns.pairplot(
            data[pair_cols],
            hue="PlacementStatus",
            diag_kind="hist"
        )

        pairplot.fig.suptitle(
            "Feature Pairplot",
            y=1.02
        )

        pairplot.fig.savefig(
            os.path.join(
                GRAPH_DIR,
                "pairplot.png"
            ),
            dpi=100,
            bbox_inches="tight"
        )

        plt.close("all")

        results["pairplot"] = (
            "/static/eda/pairplot.png"
        )


    # ========================================================
    # RETURN ALL EDA RESULTS
    # ========================================================

    return results