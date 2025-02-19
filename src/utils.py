import os
import sys
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

class GroupBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols:list, target_cols:list, strategy: str = "median"):
        """
        Custom imputer that fills missing values based on group statistics.

        Args:
            group_cols (list): List of columns to group by (e.g., ['brand', 'model', 'model_year']).
            target_cols (list): List of columns to impute (e.g., ['hp', 'liters']).
            strategy (str): "median" for numerical columns, "mode" for categorical columns.

        """
        self.group_cols = group_cols
        self.target_cols = target_cols
        self.strategy = strategy
        self.group_stats = {}  # Store computed values

    def fit(self, X:pd.DataFrame, y=None):
        """Compute group-based statistics from training data."""
        df = X.copy()

        if self.strategy == "median":
            self.group_stats = df.groupby(self.group_cols)[self.target_cols].median()
        elif self.strategy == "mode":
            self.group_stats = df.groupby(self.group_cols)[self.target_cols].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        else:
            raise ValueError("Strategy must be 'median' or 'mode'")

        # Store global fallback values (for unseen groups in test set)
        self.global_stats = df[self.target_cols].median() if self.strategy == "median" else df[self.target_cols].mode().iloc[0]

        return self  # Required for compatibility with sklearn pipeline

    def transform(self, X:pd.DataFrame):
        """Impute missing values using learned statistics."""
        df = X.copy()

        for col in self.target_cols:
            df[col] = df.apply(lambda row: self._get_imputed_value(row, col), axis=1)

        return df

    def _get_imputed_value(self, row: pd.Series, col: str):
        """Fetch the group-based imputed value, falling back to global stats if needed."""
        if pd.isnull(row[col]):  # Only impute if the value is missing
            try:
                # Try to get the group-based imputed value
                value = self.group_stats.loc[
                    (row[self.group_cols[0]], row[self.group_cols[1]], row[self.group_cols[2]]), col
                ]
                # Return the group-based value if it's not null, otherwise fallback to global stats
                return value if pd.notnull(value) else self.global_stats[col]
            except KeyError:
                # If group is not found, use global stats
                return self.global_stats[col]
        else:
            # If the value is already present (not missing), return the original value
            return row[col]
        


def evaluate_model(y, y_pred, save_path = "artifacts/figures"):
    """
    Evaluates the given model using RMSE, MSE, MAE, R² Score, and plots residuals.

    Parameters:
    model: Trained regression model
    X: Feature matrix (DataFrame or array)
    y: Target values (Series or array)
    save_path: Folder path to save the plots (default is "artifacts/figures")
    """
    try:

        # Compute evaluation metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)

        # Print the metrics
        print(f"Model Evaluation:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        # Prepare residuals
        residuals = y - y_pred

        # Create folder for saving figures if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save residuals histogram plot (overwrites each time)
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, bins=20, kde=True)
        plt.axvline(0, color='red', linestyle='dashed')
        plt.title("Residuals Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(save_path, "residuals_distribution.png"))
        plt.close()

        # Save residuals vs predicted values plot (overwrites each time)
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(0, color='red', linestyle='dashed')
        plt.title("Residuals vs. Predicted Values")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.savefig(os.path.join(save_path, "residuals_vs_predicted.png"))
        plt.close()

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        raise