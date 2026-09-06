from flask import Flask, render_template, request

from load_data import load_data, get_data_summary
from placement_eda import run_eda
from preprocessing import get_preprocessing_summary


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

        # Load CSV
        df = load_data()

        # Generate summary
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

        # Run complete EDA
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

@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing_page():

    error = None
    summary = None

    # Default scaling
    scaling_method = "standard"

    if request.method == "POST":

        scaling_method = request.form.get(
            "scaling_method",
            "standard"
        )

    try:

        # Load dataset
        df = load_data()

        # Generate preprocessing summary
        summary = get_preprocessing_summary(
            df,
            scaling_method=scaling_method
        )

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="preprocessing",
        summary=summary,
        error=error,
        scaling_method=scaling_method
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )