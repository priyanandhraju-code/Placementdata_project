from flask import Flask, render_template, request

import pandas as pd

from load_data import (
    load_data,
    get_data_summary
)

from placement_eda import run_eda

from preprocessing import preprocess_data

from linear_regression import (
    run_linear_regression
)

from logistic_regression import (
    run_logistic_regression
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        # ----------------------------------------------------
        # Load original dataset
        # ----------------------------------------------------

        df = load_data()

        # ----------------------------------------------------
        # Generate data loading summary
        # ----------------------------------------------------

        summary = get_data_summary(df)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        # ----------------------------------------------------
        # Run EDA
        # ----------------------------------------------------

        results = run_eda()

        # ----------------------------------------------------
        # Print results for debugging
        # ----------------------------------------------------

        print(results)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="eda",
        results=results,
        error=error
    )


# ============================================================
# DATA PREPROCESSING
# ============================================================

@app.route(
    "/preprocessing",
    methods=["GET", "POST"]
)
def preprocessing_page():

    error = None

    preprocessing_result = None

    # --------------------------------------------------------
    # Default scaling method
    # --------------------------------------------------------

    scaling_method = "standard"

    # ========================================================
    # GET SELECTED SCALING METHOD
    # ========================================================

    if request.method == "POST":

        scaling_method = request.form.get(
            "scaling_method",
            "standard"
        )

    try:

        # ====================================================
        # LOAD ORIGINAL DATASET
        # ====================================================

        df = load_data()

        # ====================================================
        # RUN PREPROCESSING
        # ====================================================

        preprocessing_result = preprocess_data(
            scaling_method=scaling_method
        )

        # ====================================================
        # MAKE SURE RESULT IS A DICTIONARY
        # ====================================================

        if preprocessing_result is None:

            preprocessing_result = {}

        # ====================================================
        # BASIC DATASET INFORMATION
        # ====================================================

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        preprocessing_result["target"] = "PlacementStatus"

        # ----------------------------------------------------
        # Number of features
        #
        # Total columns - target column
        # ----------------------------------------------------

        preprocessing_result["feature_count"] = int(
            len(df.columns) - 1
        )

        # ----------------------------------------------------
        # Total missing values
        # ----------------------------------------------------

        preprocessing_result["missing_total"] = int(
            df.isnull().sum().sum()
        )

        # ----------------------------------------------------
        # Duplicate rows
        # ----------------------------------------------------

        preprocessing_result["duplicate_rows"] = int(
            df.duplicated().sum()
        )

        # ====================================================
        # TARGET DISTRIBUTION
        # ====================================================

        target_distribution = (
            df["PlacementStatus"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        preprocessing_result["target_distribution"] = {

            str(status): int(count)

            for status, count
            in target_distribution.items()

        }

        # ====================================================
        # SCALING METHOD
        # ====================================================

        preprocessing_result["scaling_method"] = (
            scaling_method
        )

        # ====================================================
        # NUMERICAL FEATURES
        # ====================================================

        if not preprocessing_result.get(
            "numeric_features"
        ):

            numeric_features = (
                df.drop(
                    columns=["PlacementStatus"],
                    errors="ignore"
                )
                .select_dtypes(
                    include=["number"]
                )
                .columns
                .tolist()
            )

            preprocessing_result[
                "numeric_features"
            ] = numeric_features

        # ====================================================
        # CATEGORICAL FEATURES
        # ====================================================

        if not preprocessing_result.get(
            "categorical_features"
        ):

            categorical_features = (
                df.drop(
                    columns=["PlacementStatus"],
                    errors="ignore"
                )
                .select_dtypes(
                    exclude=["number"]
                )
                .columns
                .tolist()
            )

            preprocessing_result[
                "categorical_features"
            ] = categorical_features

        # ====================================================
        # TRAIN / TEST SIZE
        # ====================================================

        if "train_size" not in preprocessing_result:

            preprocessing_result[
                "train_size"
            ] = int(len(df) * 0.80)

        if "test_size" not in preprocessing_result:

            preprocessing_result[
                "test_size"
            ] = int(len(df) * 0.20)

        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        print()
        print("=" * 70)
        print("PLACEMENT PROJECT - PREPROCESSING")
        print("=" * 70)

        print()
        print("Target:")
        print(
            preprocessing_result["target"]
        )

        print()
        print("Number of features:")
        print(
            preprocessing_result["feature_count"]
        )

        print()
        print("Total missing values:")
        print(
            preprocessing_result["missing_total"]
        )

        print()
        print("Duplicate rows:")
        print(
            preprocessing_result["duplicate_rows"]
        )

        print()
        print("Scaling method:")
        print(
            preprocessing_result["scaling_method"]
        )

        print()
        print("Target distribution:")
        print(
            preprocessing_result[
                "target_distribution"
            ]
        )

        print()
        print("=" * 70)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    # ========================================================
    # RENDER PREPROCESSING PAGE
    # ========================================================

    return render_template(
        "index.html",
        active="preprocessing",
        summary=preprocessing_result,
        scaling_method=scaling_method,
        error=error
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

@app.route("/linear-regression")
def linear_regression_page():

    error = None

    result = None

    try:

        # ----------------------------------------------------
        # Run Linear Regression
        # ----------------------------------------------------

        result = run_linear_regression()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        linear_regression_result=result,
        error=error
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    error = None

    result = None

    try:

        # ----------------------------------------------------
        # Run Logistic Regression
        # ----------------------------------------------------

        result = run_logistic_regression()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        logistic_regression_result=result,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )