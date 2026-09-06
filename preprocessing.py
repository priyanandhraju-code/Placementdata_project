import os

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


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"placement_predict_50k Dataset (3)(in).csv"

PREPROCESSED_PATH = "preprocessed.csv"

TARGET = "PlacementStatus"


# ============================================================
# CREATE ONE-HOT ENCODER
# ============================================================

def create_one_hot_encoder():
    """
    Create OneHotEncoder while supporting both newer
    and older versions of scikit-learn.
    """

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ============================================================
# CREATE SCALER
# ============================================================

def create_scaler(scaling_method="standard"):
    """
    Return the scaler selected by the user.

    standard -> StandardScaler
    minmax   -> MinMaxScaler
    """

    scaling_method = scaling_method.lower()

    if scaling_method == "minmax":

        return MinMaxScaler()

    elif scaling_method == "standard":

        return StandardScaler()

    else:

        raise ValueError(
            "Invalid scaling method. "
            "Use 'standard' or 'minmax'."
        )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    scaling_method="standard"
):
    """
    Separate features and target and create the
    preprocessing transformer.
    """

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' "
            f"not found in dataset"
        )

    data = df.copy()

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = data.drop(
        columns=[TARGET]
    )

    y = data[TARGET]

    # --------------------------------------------------------
    # Identify numerical and categorical columns
    # --------------------------------------------------------

    numeric_cols = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_cols = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # Numerical preprocessing
    #
    # 1. Fill missing values with median
    # 2. Apply selected scaling method
    # --------------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            create_scaler(
                scaling_method
            )
        )
    ])

    # --------------------------------------------------------
    # Categorical preprocessing
    #
    # 1. Fill missing values with most frequent value
    # 2. Convert categories to numerical columns
    # --------------------------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            create_one_hot_encoder()
        )
    ])

    # --------------------------------------------------------
    # Combine numerical and categorical preprocessing
    # --------------------------------------------------------

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


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(
    df: pd.DataFrame,
    scaling_method="standard",
    test_size=0.2,
    random_state=42
):
    """
    Split data into training and testing sets.

    The transformer is returned so that model-training
    modules can fit preprocessing only on the training data.
    """

    (
        X,
        y,
        transformer,
        numeric_cols,
        categorical_cols
    ) = prepare_data(
        df,
        scaling_method
    )

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

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
        "categorical_cols": categorical_cols
    }


# ============================================================
# GENERATE PREPROCESSED DATASET
# ============================================================

def generate_preprocessed_csv(
    scaling_method="standard",
    input_path=DATA_PATH,
    output_path=PREPROCESSED_PATH
):
    """
    Read the original dataset, perform preprocessing,
    and save the transformed dataset as preprocessed.csv.

    This function is intended for creating the complete
    preprocessed dataset for inspection/export.
    """

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        input_path
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        X,
        y,
        transformer,
        numeric_cols,
        categorical_cols
    ) = prepare_data(
        df,
        scaling_method
    )

    # --------------------------------------------------------
    # Fit transformer and transform features
    # --------------------------------------------------------

    X_processed = transformer.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        transformer.get_feature_names_out()
    )

    # --------------------------------------------------------
    # Convert transformed data to DataFrame
    # --------------------------------------------------------

    processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=df.index
    )

    # --------------------------------------------------------
    # Add target column back
    # --------------------------------------------------------

    processed_df[TARGET] = y.values

    # --------------------------------------------------------
    # Save preprocessed dataset
    # --------------------------------------------------------

    processed_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Return useful information
    # --------------------------------------------------------

    return {
        "output_path": output_path,
        "original_rows": int(df.shape[0]),
        "original_columns": int(df.shape[1]),
        "processed_rows": int(processed_df.shape[0]),
        "processed_columns": int(processed_df.shape[1]),
        "scaling_method": scaling_method,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "processed_features": list(
            processed_df.columns
        )
    }


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

def get_preprocessing_summary(
    df: pd.DataFrame,
    scaling_method="standard"
):
    """
    Return preprocessing information for the Flask UI.
    """

    (
        X,
        y,
        transformer,
        numeric_cols,
        categorical_cols
    ) = prepare_data(
        df,
        scaling_method
    )

    return {
        "target": TARGET,

        "feature_count": int(
            X.shape[1]
        ),

        "numeric_features": numeric_cols,

        "categorical_features": categorical_cols,

        "target_distribution": {
            str(key): int(value)
            for key, value in (
                y.value_counts()
                .sort_index()
                .items()
            )
        },

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "missing_total": int(
            df.isnull().sum().sum()
        ),

        "scaling_method": scaling_method
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("PLACEMENT DATA PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Choose scaling method
    # --------------------------------------------------------
    #
    # Change this to:
    #
    # "standard"
    #
    # or:
    #
    # "minmax"
    #
    # --------------------------------------------------------

    scaling_method = "standard"

    print()
    print(
        f"Scaling method: {scaling_method}"
    )

    print()
    print(
        "Reading original dataset..."
    )

    try:

        result = generate_preprocessed_csv(
            scaling_method=scaling_method
        )

        print()
        print(
            "Preprocessing completed successfully."
        )

        print()
        print(
            f"Original rows: "
            f"{result['original_rows']}"
        )

        print(
            f"Original columns: "
            f"{result['original_columns']}"
        )

        print(
            f"Processed rows: "
            f"{result['processed_rows']}"
        )

        print(
            f"Processed columns: "
            f"{result['processed_columns']}"
        )

        print()
        print(
            f"Scaling method: "
            f"{result['scaling_method']}"
        )

        print()
        print(
            f"Created file: "
            f"{result['output_path']}"
        )

        print()
        print("=" * 60)

    except Exception as e:

        print()
        print(
            "Preprocessing failed."
        )

        print(
            f"Error: {e}"
        )

        print()
        print("=" * 60)