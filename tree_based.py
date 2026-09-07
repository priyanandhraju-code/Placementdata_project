import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = "preprocessed.csv"

TARGET_COLUMN = "PlacementStatus"

TEST_SIZE = 0.20

RANDOM_STATE = 42

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# LOAD XGBOOST / LIGHTGBM WHEN REQUIRED
# ============================================================

def get_xgb_classifier():

    try:

        from xgboost import XGBClassifier

        return XGBClassifier

    except ImportError:

        raise ImportError(
            "XGBoost is not installed. "
            "Run: pip install xgboost"
        )


def get_lgbm_classifier():

    try:

        from lightgbm import LGBMClassifier

        return LGBMClassifier

    except ImportError:

        raise ImportError(
            "LightGBM is not installed. "
            "Run: pip install lightgbm"
        )


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(

            f"Preprocessed dataset not found: "
            f"{FILE_PATH}"

        )

    df = pd.read_csv(
        FILE_PATH
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # --------------------------------------------------------
    # Check target column
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        raise ValueError(

            f"Target column "
            f"'{TARGET_COLUMN}' not found "
            f"in preprocessed dataset."

        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Make sure features are numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # Convert target to integer
    # --------------------------------------------------------

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    y = y.fillna(
        0
    )

    y = y.astype("int64")

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    # --------------------------------------------------------
    # Class predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        predictions

    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(

        y_test,

        predictions,

        zero_division=0

    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(

        y_test,

        predictions,

        zero_division=0

    )

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1 = f1_score(

        y_test,

        predictions,

        zero_division=0

    )

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(

        y_test,

        probabilities

    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc_auc,

        "predictions": predictions,

        "probabilities": probabilities

    }


# ============================================================
# 1. DECISION TREE
# ============================================================

def run_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\nRunning Decision Tree..."
    )

    model = DecisionTreeClassifier(

        random_state=RANDOM_STATE,

        class_weight="balanced",

        max_depth=12,

        min_samples_leaf=10

    )

    model.fit(

        X_train,

        y_train

    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    return model, metrics


# ============================================================
# 2. RANDOM FOREST
# ============================================================

def run_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\nRunning Random Forest..."
    )

    model = RandomForestClassifier(

        n_estimators=100,

        random_state=RANDOM_STATE,

        class_weight="balanced",

        n_jobs=-1,

        max_depth=15,

        min_samples_leaf=5

    )

    model.fit(

        X_train,

        y_train

    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    return model, metrics


# ============================================================
# 3. EXTRA TREES
# ============================================================

def run_extra_trees(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\nRunning Extra Trees..."
    )

    model = ExtraTreesClassifier(

        n_estimators=100,

        random_state=RANDOM_STATE,

        class_weight="balanced",

        n_jobs=-1,

        max_depth=15,

        min_samples_leaf=5

    )

    model.fit(

        X_train,

        y_train

    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    return model, metrics


# ============================================================
# 4. GRADIENT BOOSTING
# ============================================================

def run_gradient_boosting(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\nRunning Gradient Boosting..."
    )

    model = GradientBoostingClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=3,

        random_state=RANDOM_STATE

    )

    model.fit(

        X_train,

        y_train

    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    return model, metrics


# ============================================================
# 5. ADABOOST
# ============================================================

def run_adaboost(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\nRunning AdaBoost..."
    )

    model = AdaBoostClassifier(

        n_estimators=100,

        learning_rate=0.50,

        random_state=RANDOM_STATE

    )

    model.fit(

        X_train,

        y_train

    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    return model, metrics


# ============================================================
# 6. XGBOOST
# ============================================================

def run_xgboost(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\nRunning XGBoost..."
    )

    XGBClassifier = get_xgb_classifier()

    model = XGBClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        eval_metric="logloss",

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train

    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    return model, metrics


# ============================================================
# 7. LIGHTGBM
# ============================================================

def run_lightgbm(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "\nRunning LightGBM..."
    )

    LGBMClassifier = get_lgbm_classifier()

    model = LGBMClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=-1,

        num_leaves=31,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1

    )

    model.fit(

        X_train,

        y_train

    )

    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )

    return model, metrics


# ============================================================
# ALGORITHM MAP
# ============================================================

ALGORITHMS = {

    "decision_tree": (

        "Decision Tree",

        run_decision_tree

    ),

    "random_forest": (

        "Random Forest",

        run_random_forest

    ),

    "extra_trees": (

        "Extra Trees",

        run_extra_trees

    ),

    "gradient_boosting": (

        "Gradient Boosting",

        run_gradient_boosting

    ),

    "adaboost": (

        "AdaBoost",

        run_adaboost

    ),

    "xgboost": (

        "XGBoost",

        run_xgboost

    ),

    "lightgbm": (

        "LightGBM",

        run_lightgbm

    )

}


# ============================================================
# RUN SELECTED TREE ALGORITHM
# ============================================================

def run_tree_algorithm(
    algorithm
):

    # --------------------------------------------------------
    # Validate algorithm
    # --------------------------------------------------------

    if algorithm not in ALGORITHMS:

        raise ValueError(

            f"Unknown tree algorithm: "
            f"{algorithm}"

        )

    # --------------------------------------------------------
    # Load preprocessed data
    # --------------------------------------------------------

    df = load_preprocessed_data()

    print()
    print("=" * 80)
    print(
        "PLACEMENT PROJECT - V5 TREE BASED MODELS"
    )
    print("=" * 80)

    print()
    print("Selected Algorithm:")

    print(
        ALGORITHMS[algorithm][0]
    )

    print()
    print("Dataset Shape:")

    print(
        df.shape
    )

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X, y = prepare_data(
        df
    )

    print()
    print("Number of Features:")

    print(
        X.shape[1]
    )

    print()
    print("Target:")

    print(
        TARGET_COLUMN
    )

    print()
    print("Target Distribution:")

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # Check numerical data
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FEATURE MATRIX NUMERICAL CHECK"
    )
    print("=" * 80)

    print()
    print("Feature Data Types:")

    print(
        X.dtypes.value_counts()
    )

    print()
    print("NaN Values:")

    print(
        X.isna().sum().sum()
    )

    print()
    print("Infinite Values:")

    print(
        X.isin(
            [float("inf"), float("-inf")]
        ).sum().sum()
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(

        X,

        y

    )

    print()
    print("Training Shape:")

    print(
        X_train.shape
    )

    print()
    print("Testing Shape:")

    print(
        X_test.shape
    )

    # --------------------------------------------------------
    # Select algorithm
    # --------------------------------------------------------

    display_name, function = ALGORITHMS[algorithm]

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    model, metrics = function(

        X_train,

        X_test,

        y_train,

        y_test

    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        display_name.upper()
    )

    print("=" * 80)

    print(
        "Accuracy:",
        metrics["Accuracy"]
    )

    print(
        "Precision:",
        metrics["Precision"]
    )

    print(
        "Recall:",
        metrics["Recall"]
    )

    print(
        "F1 Score:",
        metrics["F1 Score"]
    )

    print(
        "ROC AUC:",
        metrics["ROC AUC"]
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "algorithm": display_name,

        "dataset_shape": df.shape,

        "feature_count": X.shape[1],

        "training_shape": X_train.shape,

        "testing_shape": X_test.shape,

        "Accuracy": metrics["Accuracy"],

        "Precision": metrics["Precision"],

        "Recall": metrics["Recall"],

        "F1 Score": metrics["F1 Score"],

        "ROC AUC": metrics["ROC AUC"],

        "model": model,

        "X_test": X_test,

        "y_test": y_test,

        "predictions": metrics["predictions"],

        "probabilities": metrics["probabilities"]

    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_plot(
    result,
    filename
):

    os.makedirs(

        PLOT_DIRECTORY,

        exist_ok=True

    )

    # --------------------------------------------------------
    # Calculate confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(

        result["y_test"],

        result["predictions"]

    )

    # --------------------------------------------------------
    # Create display
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(

        confusion_matrix=matrix

    )

    display.plot()

    plt.title(

        f'{result["algorithm"]} - Confusion Matrix'

    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    path = os.path.join(

        PLOT_DIRECTORY,

        filename

    )

    plt.savefig(

        path,

        dpi=150

    )

    plt.close()

    return path


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve_plot(
    result,
    filename
):

    os.makedirs(

        PLOT_DIRECTORY,

        exist_ok=True

    )

    # --------------------------------------------------------
    # Calculate ROC curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(

        result["y_test"],

        result["probabilities"]

    )

    # --------------------------------------------------------
    # Create ROC plot
    # --------------------------------------------------------

    plt.figure(

        figsize=(8, 6)

    )

    plt.plot(

        fpr,

        tpr,

        label=(

            f'ROC AUC = '
            f'{result["ROC AUC"]:.4f}'

        )

    )

    # --------------------------------------------------------
    # Random classifier reference line
    # --------------------------------------------------------

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(

        f'{result["algorithm"]} - ROC Curve'

    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    path = os.path.join(

        PLOT_DIRECTORY,

        filename

    )

    plt.savefig(

        path,

        dpi=150

    )

    plt.close()

    return path