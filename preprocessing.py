import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Target variable
TARGET = "PlacementStatus"


def prepare_data(df: pd.DataFrame, scaling_method="standard"):
    """
    Prepare the dataset for machine learning.

    Preprocessing steps:
    1. Separate features and target
    2. Identify numerical and categorical columns
    3. Handle missing values
    4. Scale numerical features
    5. Encode categorical features
    """

    # --------------------------------------------------
    # Check target column
    # --------------------------------------------------

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in dataset"
        )

    # Make a copy so the original DataFrame is not modified
    data = df.copy()

    # --------------------------------------------------
    # Step 1: Feature / Target Separation
    # --------------------------------------------------

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    # --------------------------------------------------
    # Step 2: Identify Numerical and Categorical Columns
    # --------------------------------------------------

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # --------------------------------------------------
    # Step 3: Select Scaling Method
    # --------------------------------------------------

    scaling_method = scaling_method.lower()

    if scaling_method == "standard":
        scaler = StandardScaler()

    elif scaling_method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError(
            "Invalid scaling method. "
            "Choose either 'standard' or 'minmax'."
        )

    # --------------------------------------------------
    # Step 4: Numerical Pipeline
    # --------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            scaler
        )
    ])

    # --------------------------------------------------
    # Step 5: Categorical Pipeline
    # --------------------------------------------------

    # Compatibility with different scikit-learn versions
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            encoder
        )
    ])

    # --------------------------------------------------
    # Step 6: Combine Numerical + Categorical Pipelines
    # --------------------------------------------------

    transformer = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_cols
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    ])

    return (
        X,
        y,
        transformer,
        numeric_cols,
        categorical_cols
    )


def split_data(
    df: pd.DataFrame,
    scaling_method="standard",
    test_size=0.2,
    random_state=42
):
    """
    Split the dataset into training and testing data.

    Default:
    80% Training
    20% Testing
    """

    (
        X,
        y,
        transformer,
        numeric_cols,
        categorical_cols
    ) = prepare_data(
        df,
        scaling_method=scaling_method
    )

    # --------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "transformer": transformer,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "scaling_method": scaling_method
    }


def get_preprocessing_summary(
    df: pd.DataFrame,
    scaling_method="standard"
):
    """
    Generate preprocessing information
    for displaying in the Flask UI.
    """

    (
        X,
        y,
        _,
        numeric_cols,
        categorical_cols
    ) = prepare_data(
        df,
        scaling_method=scaling_method
    )

    # Calculate target distribution
    target_distribution = {
        str(key): int(value)
        for key, value in
        y.value_counts().sort_index().items()
    }

    # Calculate train/test sizes
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size

    return {
        "target": TARGET,

        "feature_count": int(
            X.shape[1]
        ),

        "numeric_features": numeric_cols,

        "categorical_features": categorical_cols,

        "target_distribution":
            target_distribution,

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "missing_total": int(
            df.isnull().sum().sum()
        ),

        "scaling_method":
            scaling_method,

        "train_size":
            train_size,

        "test_size":
            test_size
    }