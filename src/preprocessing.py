from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode string categorical features
    cat_cols = working_train_df.select_dtypes(include=["object"]).columns
    binary_cols = [col for col in cat_cols if working_train_df[col].nunique() == 2]
    multi_cols = [col for col in cat_cols if working_train_df[col].nunique() > 2]

    # Binary encoding
    ord_encoder = OrdinalEncoder()
    working_train_df[binary_cols] = ord_encoder.fit_transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ord_encoder.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ord_encoder.transform(working_test_df[binary_cols])

    # One-hot encoding
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot_encoder.fit(working_train_df[multi_cols])

    def apply_onehot(df, encoder, cols):
        onehot_array = encoder.transform(df[cols])
        onehot_df = pd.DataFrame(
            onehot_array,
            columns=encoder.get_feature_names_out(cols),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=cols), onehot_df], axis=1)
        return df

    working_train_df = apply_onehot(working_train_df, onehot_encoder, multi_cols)
    working_val_df = apply_onehot(working_val_df, onehot_encoder, multi_cols)
    working_test_df = apply_onehot(working_test_df, onehot_encoder, multi_cols)

    # Align columns before imputation
    train_cols = working_train_df.columns
    working_val_df = working_val_df.reindex(columns=train_cols, fill_value=0)
    working_test_df = working_test_df.reindex(columns=train_cols, fill_value=0)
    
    # 3. Impute missing values with the median
    imputer = SimpleImputer(strategy="median")
    imputer.fit(working_train_df)

    working_train_df = pd.DataFrame(imputer.transform(working_train_df), columns=train_cols)
    working_val_df = pd.DataFrame(imputer.transform(working_val_df), columns=train_cols)
    working_test_df = pd.DataFrame(imputer.transform(working_test_df), columns=train_cols)

    for df in [working_train_df, working_val_df, working_test_df]:
        df.fillna(0, inplace=True)
    # Remove constant columns (features with no variance) from the training set
    # This must be done BEFORE scaling to avoid division by zero
    constant_cols = [
        col for col in working_train_df.columns if working_train_df[col].nunique() == 1
    ]
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        working_train_df.drop(columns=constant_cols, inplace=True)
        working_val_df.drop(columns=constant_cols, inplace=True)
        working_test_df.drop(columns=constant_cols, inplace=True)

    # 5. Feature scaling with Min-Max scaler
    scaler = MinMaxScaler()
    scaler.fit(working_train_df)
    
    # Get the final list of columns after removal
    working_train_df = pd.DataFrame(scaler.transform(working_train_df), columns=scaler.feature_names_in_)
    working_val_df = pd.DataFrame(scaler.transform(working_val_df), columns=scaler.feature_names_in_)
    working_test_df = pd.DataFrame(scaler.transform(working_test_df), columns=scaler.feature_names_in_)

    # Double-check no NaNs remain
    assert not np.isnan(working_train_df.to_numpy()).any(), "NaN detected in train data"
    assert not np.isnan(working_val_df.to_numpy()).any(), "NaN detected in val data"
    assert not np.isnan(working_test_df.to_numpy()).any(), "NaN detected in test data"

    return (
        working_train_df.to_numpy(),
        working_val_df.to_numpy(),
        working_test_df.to_numpy(),
    )