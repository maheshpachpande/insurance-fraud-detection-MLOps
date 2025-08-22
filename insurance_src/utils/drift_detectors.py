
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
from insurance_src.logger import logging
import math


class DriftDetectionResult:
    """Encapsulates dataset drift detection result for a feature."""
    def __init__(self, p_value: float, drift_detected: bool):
        self.p_value = p_value
        self.drift_detected = drift_detected


class DriftDetector:
    """Detects dataset, prior probability, and concept drift between two datasets."""

    def __init__(self):
        pass

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, DriftDetectionResult]]:
        """
        Detects drift for each column automatically based on its data type.

        Numeric columns: KS-test
        Categorical columns: LabelEncoded then KS-test
        """
        overall_status = True
        drift_report: Dict[str, DriftDetectionResult] = {}
        shared_cols = base_df.columns.intersection(current_df.columns)

        for column in shared_cols:
            series_base = base_df[column].dropna()
            series_current = current_df[column].dropna()

            if series_base.empty or series_current.empty:
                logging.warning("Column '%s' contains only NA values in one dataset.", column)
                drift_report[column] = DriftDetectionResult(p_value=math.nan, drift_detected=True)
                overall_status = False
                continue

            # If categorical, label encode first
            if pd.api.types.is_object_dtype(series_base) or pd.api.types.is_object_dtype(series_current):
                le = LabelEncoder()
                combined = pd.concat([series_base.astype(str), series_current.astype(str)])
                le.fit(combined)
                series_base = le.transform(series_base.astype(str))
                series_current = le.transform(series_current.astype(str))

            # KS test for all numeric arrays
            result = ks_2samp(series_base, series_current)
            p_value = result.pvalue # type: ignore
            drift_detected = p_value < threshold

            drift_report[column] = DriftDetectionResult(p_value=p_value, drift_detected=drift_detected)
            if drift_detected:
                logging.info("Drift detected in column '%s' (p=%.4f)", column, p_value)
                overall_status = False

        return overall_status, drift_report

    def detect_prior_probability_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compares class distribution probabilities between two dataframes for a target column.

        :return: Dict[class, {"base_prob", "current_prob", "abs_diff"}]
        """
        base_dist = base_df[target_col].value_counts(normalize=True)
        curr_dist = current_df[target_col].value_counts(normalize=True)
        all_classes = set(base_dist.index).union(curr_dist.index)
        report = {}
        for cls in all_classes:
            base_prob = float(base_dist.get(cls, 0.0))
            current_prob = float(curr_dist.get(cls, 0.0))
            abs_diff = abs(base_prob - current_prob)
            report[str(cls)] = {
                "base_prob": base_prob,
                "current_prob": current_prob,
                "abs_diff": abs_diff
            }
            logging.info(
                "Class '%s': base=%.4f, current=%.4f, abs_diff=%.4f", cls, base_prob, current_prob, abs_diff
            )
        return report

    def detect_concept_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str
    ) -> float:
        """
        Trains a model (logistic regression) on base data and tests on current data to estimate target prediction accuracy.

        :return: accuracy_score
        """
        base_df_copy = base_df.copy()
        current_df_copy = current_df.copy()

        # Encode target
        le = LabelEncoder()
        base_df_copy[target_col] = le.fit_transform(base_df_copy[target_col])
        current_df_copy[target_col] = le.transform(current_df_copy[target_col])
        X_train = base_df_copy.drop(columns=[target_col])
        y_train = base_df_copy[target_col]
        X_test = current_df_copy.drop(columns=[target_col])
        y_test = current_df_copy[target_col]

        # Encode categorical features
        for column in X_train.columns:
            if pd.api.types.is_object_dtype(X_train[column]) or pd.api.types.is_object_dtype(X_test[column]):
                le_col = LabelEncoder()
                combined = pd.concat([X_train[column], X_test[column]]).astype(str)
                le_col.fit(combined)
                X_train[column] = le_col.transform(X_train[column].astype(str))
                X_test[column] = le_col.transform(X_test[column].astype(str))

        # Standardize numeric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit model, evaluate accuracy
        model = LogisticRegression(max_iter=200_000, solver="saga", n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

        logging.info("Concept drift (accuracy on current): %.4f", accuracy)
        return float(accuracy)


if __name__ == "__main__":
    
    # Dummy example - replace with your dataframes and target column
    base = pd.read_csv("artifact/data_ingestion/ingested/train.csv")
    curr = pd.read_csv("artifact/data_ingestion/ingested/test.csv")
    detector = DriftDetector()
    _, drift_report = detector.detect_dataset_drift(base, curr)
    for k, v in drift_report.items():
        print(f"{k}: p={v.p_value:.3f}, drift={v.drift_detected}")
 
    print(len(drift_report.items()))
