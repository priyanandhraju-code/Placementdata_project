import os
import pandas as pd

# Dataset Path
DATA_PATH = r"placement_predict_50k Dataset (3)(in).csv"


# Load Dataset
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    return df


# Dataset Summary
def get_data_summary(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {
            col: int(df[col].isnull().sum()) for col in df.columns
        },
        "preview": df.head(10).to_dict(orient="records"),
    }

    return summary