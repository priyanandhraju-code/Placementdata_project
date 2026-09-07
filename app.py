from flask import Flask, render_template, request

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

from tree_based import (
    run_tree_algorithm,
    create_confusion_matrix_plot,
    create_roc_curve_plot
)


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

        df = load_data()

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
# EDA
# ============================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        results = run_eda()

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
# PREPROCESSING
# ============================================================

@app.route(
    "/preprocessing",
    methods=["GET", "POST"]
)
def preprocessing_page():

    error = None
    preprocessing_result = None

    scaling_method = "standard"

    if request.method == "POST":

        scaling_method = request.form.get(
            "scaling_method",
            "standard"
        )

    try:

        # ----------------------------------------------------
        # RUN PREPROCESSING
        # ----------------------------------------------------

        preprocessing_result = preprocess_data(
            scaling_method=scaling_method
        )


        # ----------------------------------------------------
        # LOAD ORIGINAL DATA
        # ----------------------------------------------------

        df = load_data()


        # ----------------------------------------------------
        # BUILD DISPLAY INFORMATION
        # ----------------------------------------------------

        target_column = "PlacementStatus"


        # Features BEFORE target separation
        feature_columns = [
            column
            for column in df.columns
            if column != target_column
        ]


        # ----------------------------------------------------
        # NUMERICAL FEATURES
        # ----------------------------------------------------

        numeric_features = df[
            feature_columns
        ].select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()


        # ----------------------------------------------------
        # CATEGORICAL FEATURES
        # ----------------------------------------------------

        categorical_features = df[
            feature_columns
        ].select_dtypes(
            include=["object", "category"]
        ).columns.tolist()


        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        missing_total = int(
            df[feature_columns]
            .isnull()
            .sum()
            .sum()
        )


        # ----------------------------------------------------
        # DUPLICATE ROWS
        # ----------------------------------------------------

        duplicate_rows = int(
            df.duplicated().sum()
        )


        # ----------------------------------------------------
        # TARGET DISTRIBUTION
        # ----------------------------------------------------

        target_distribution = (
            df[target_column]
            .value_counts()
            .sort_index()
            .to_dict()
        )


        target_distribution = {

            str(status): int(count)

            for status, count
            in target_distribution.items()

        }


        # ----------------------------------------------------
        # TRAIN / TEST SIZE
        # ----------------------------------------------------

        total_rows = len(df)

        train_size = int(
            total_rows * 0.80
        )

        test_size = (
            total_rows - train_size
        )


        # ----------------------------------------------------
        # ENSURE RESULT DICTIONARY EXISTS
        # ----------------------------------------------------

        if preprocessing_result is None:

            preprocessing_result = {}


        # ----------------------------------------------------
        # ADD ALL DISPLAY VALUES
        # ----------------------------------------------------

        preprocessing_result["target"] = (
            target_column
        )

        preprocessing_result["feature_count"] = (
            len(feature_columns)
        )

        preprocessing_result["missing_total"] = (
            missing_total
        )

        preprocessing_result["duplicate_rows"] = (
            duplicate_rows
        )

        preprocessing_result["scaling_method"] = (
            scaling_method
        )

        preprocessing_result["numeric_features"] = (
            numeric_features
        )

        preprocessing_result["categorical_features"] = (
            categorical_features
        )

        preprocessing_result["train_size"] = (
            train_size
        )

        preprocessing_result["test_size"] = (
            test_size
        )

        preprocessing_result["target_distribution"] = (
            target_distribution
        )


    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"


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
# TREE BASED MODELS
# ============================================================

@app.route(
    "/tree-based",
    methods=["GET", "POST"]
)
def tree_based_page():

    error = None
    result = None

    selected_algorithm = "decision_tree"


    if request.method == "POST":

        selected_algorithm = request.form.get(
            "algorithm",
            "decision_tree"
        )

        try:

            # ------------------------------------------------
            # RUN SELECTED ALGORITHM
            # ------------------------------------------------

            result = run_tree_algorithm(
                selected_algorithm
            )


            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            confusion_filename = (
                f"{selected_algorithm}"
                "_confusion_matrix.png"
            )


            create_confusion_matrix_plot(
                result,
                confusion_filename
            )


            # ------------------------------------------------
            # ROC CURVE
            # ------------------------------------------------

            roc_filename = (
                f"{selected_algorithm}"
                "_roc_curve.png"
            )


            create_roc_curve_plot(
                result,
                roc_filename
            )


            # ------------------------------------------------
            # STORE PLOT FILENAMES
            # ------------------------------------------------

            result[
                "confusion_matrix_plot"
            ] = confusion_filename


            result[
                "roc_curve_plot"
            ] = roc_filename


        except FileNotFoundError as e:

            error = str(e)

        except ImportError as e:

            error = str(e)

        except Exception as e:

            error = f"Unexpected error: {e}"


    return render_template(
        "tree_based.html",
        active="tree-based",
        selected_algorithm=selected_algorithm,
        result=result,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )