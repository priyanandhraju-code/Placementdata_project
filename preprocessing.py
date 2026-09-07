import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"placement_predict_50k Dataset (3)(in).csv"

PREPROCESSED_PATH = r"preprocessed.csv"

TARGET_COLUMN = "PlacementStatus"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# CREATE SCALER
# ============================================================

def create_scaler(method="standard"):

    if method == "minmax":

        return MinMaxScaler()

    return StandardScaler()


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(scaling_method="standard"):

    df = load_dataset()

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    # --------------------------------------------------------
    # Separate target and features
    # --------------------------------------------------------

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Identify numerical and categorical columns
    # --------------------------------------------------------

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Handle missing numerical values
    # --------------------------------------------------------

    if numerical_columns:

        X[numerical_columns] = X[numerical_columns].fillna(
            X[numerical_columns].median()
        )

    # --------------------------------------------------------
    # Handle missing categorical values
    # --------------------------------------------------------

    for column in categorical_columns:

        if X[column].isnull().any():

            X[column] = X[column].fillna(
                X[column].mode()[0]
            )

    # --------------------------------------------------------
    # Encode categorical variables
    # --------------------------------------------------------

    if categorical_columns:

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

        encoded_array = encoder.fit_transform(
            X[categorical_columns]
        )

        encoded_columns = encoder.get_feature_names_out(
            categorical_columns
        )

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoded_columns,
            index=X.index
        )

        X = X.drop(
            columns=categorical_columns
        )

        X = pd.concat(
            [X, encoded_df],
            axis=1
        )

    # --------------------------------------------------------
    # Scale numerical features
    # --------------------------------------------------------

    if numerical_columns:

        scaler = create_scaler(
            scaling_method
        )

        X[numerical_columns] = scaler.fit_transform(
            X[numerical_columns]
        )

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    if y.dtype == "object" or str(y.dtype) == "category":

        unique_values = y.dropna().unique()

        if len(unique_values) == 2:

            # Convert binary target to 0/1
            mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }

            y = y.map(mapping)

        else:

            y = pd.factorize(y)[0]

    # --------------------------------------------------------
    # Combine processed features + target
    # --------------------------------------------------------

    processed_df = X.copy()

    processed_df[TARGET_COLUMN] = y.values

    # --------------------------------------------------------
    # Save preprocessed dataset
    # --------------------------------------------------------

    processed_df.to_csv(
        PREPROCESSED_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Return information to Flask
    # --------------------------------------------------------

    return {
        "original_shape": df.shape,

        "processed_shape": processed_df.shape,

        "target_column": TARGET_COLUMN,

        "numerical_features": numerical_columns,

        "categorical_features": categorical_columns,

        "total_features_after_encoding": X.shape[1],

        "scaling_method": scaling_method,

        "output_file": PREPROCESSED_PATH,

        "missing_values_before": int(
            df.isnull().sum().sum()
        ),

        "missing_values_after": int(
            processed_df.isnull().sum().sum()
        ),

        "preview": processed_df.head(10).to_dict(
            orient="records"
        )
    }


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(test_size=0.20, random_state=42):

    if not os.path.exists(PREPROCESSED_PATH):

        preprocess_data()

    df = pd.read_csv(
        PREPROCESSED_PATH
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

def get_preprocessing_summary():

    df = load_dataset()

    processed_df = preprocess_data()

    return {
        "original_rows": df.shape[0],

        "original_columns": df.shape[1],

        "processed_rows": processed_df["processed_shape"][0],

        "processed_columns": processed_df["processed_shape"][1],

        "target": TARGET_COLUMN,

        "output_file": PREPROCESSED_PATH
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = preprocess_data()

    print("\nPreprocessing completed successfully.")

    print(
        "Original shape:",
        result["original_shape"]
    )

    print(
        "Processed shape:",
        result["processed_shape"]
    )

    print(
        "Preprocessed file:",
        result["output_file"]
    )