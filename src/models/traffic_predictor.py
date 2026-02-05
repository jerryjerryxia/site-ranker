"""
ML-based traffic tier prediction model.

This module provides a trained classifier that predicts site traffic tiers
based on GTR enforcement signals, addressing the gap where most sites lack
direct traffic measurements (Tranco rank).

Supports both:
- 5-tier system: Tier 1 (>10M) to Tier 5 (<10K)
- 3-tier system: Tier 1 (>1M), Tier 2 (10K-1M), Tier 3 (<10K)

Training data: SimilarWeb ground truth (68,851 domains) with time-aligned GTR features.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Tier descriptions
TIER_3_DESCRIPTIONS = {
    1: "High Impact (>1M visits)",
    2: "Moderate (10K-1M visits)",
    3: "Low Impact (<10K visits)",
}

TIER_5_DESCRIPTIONS = {
    1: "Tier 1 (>10M visits)",
    2: "Tier 2 (1M-10M visits)",
    3: "Tier 3 (100K-1M visits)",
    4: "Tier 4 (10K-100K visits)",
    5: "Tier 5 (<10K visits)",
}


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    adjacent_accuracy: float  # Within 1 tier
    confusion_matrix: np.ndarray
    classification_report: str


# Features used for training (must match train_3tier_models.py)
GTR_FEATURES = [
    "total_requests",
    "total_urls_removed",
    "total_urls_not_indexed",
    "total_urls_no_action",
    "total_urls_targeted",
    "removal_rate",
    "avg_urls_per_request",
    "enforcement_duration_days",
    "days_since_last_request",
    "requests_per_month",
    "major_org_requests",
    "unique_major_orgs",
    "major_org_ratio",
    "major_studio_requests",
    "unique_major_studios",
    "major_studio_ratio",
    "requests_last_30d",
    "requests_last_90d",
    "major_org_requests_last_30d",
    "major_org_requests_last_90d",
]

# Log-transform features (heavy-tailed distributions)
LOG_TRANSFORM_FEATURES = [
    "total_requests",
    "total_urls_removed",
    "total_urls_targeted",
    "avg_urls_per_request",
    "enforcement_duration_days",
    "requests_per_month",
    "major_org_requests",
    "major_studio_requests",
    "requests_last_30d",
    "requests_last_90d",
    "major_org_requests_last_30d",
    "major_org_requests_last_90d",
]


class TrafficTierPredictor:
    """
    Predicts traffic tier for domains based on GTR enforcement signals.

    Supports both 3-tier and 5-tier systems. The 3-tier system is recommended
    for production use as it provides better class balance and higher accuracy.

    3-tier system:
    - Tier 1: High Impact (>1M visits)
    - Tier 2: Moderate (10K-1M visits)
    - Tier 3: Low Impact (<10K visits)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_3tier: bool = True,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model. If None, uses default location.
            use_3tier: Whether to use 3-tier model (default) or 5-tier model.
        """
        self.model = None
        self.scaler = None
        self.feature_names = GTR_FEATURES
        self.use_3tier = use_3tier
        self.num_tiers = 3 if use_3tier else 5
        self.tier_descriptions = TIER_3_DESCRIPTIONS if use_3tier else TIER_5_DESCRIPTIONS

        # Default model paths
        if model_path is not None:
            self.model_path = model_path
        elif use_3tier:
            self.model_path = MODELS_DIR / "3tier_rf_balanced.pkl"
        else:
            self.model_path = MODELS_DIR / "traffic_tier_model.pkl"

    def load(self) -> bool:
        """
        Load trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not self.model_path.exists():
            return False

        try:
            # Try joblib first (newer models)
            saved = joblib.load(self.model_path)
            self.model = saved["model"]
            self.scaler = saved["scaler"]
            self.feature_names = saved.get("feature_cols", saved.get("feature_names", GTR_FEATURES))

            # Check for tier mapping (3-tier models have this)
            if "tier_mapping" in saved:
                self.tier_descriptions = saved["tier_mapping"]
                self.num_tiers = len(self.tier_descriptions)
                self.use_3tier = self.num_tiers == 3

            return True
        except Exception:
            # Fall back to pickle for older models
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                    self.model = saved["model"]
                    self.scaler = saved["scaler"]
                    self.feature_names = saved.get("feature_names", GTR_FEATURES)
                return True
            except Exception:
                return False

    def save(self) -> None:
        """Save trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_names,
                "tier_mapping": self.tier_descriptions,
            },
            self.model_path,
        )

    def predict(self, features: dict) -> tuple[int, float]:
        """
        Predict traffic tier for a single domain.

        Args:
            features: Dictionary of GTR features for the domain.

        Returns:
            Tuple of (tier, confidence) where tier is 1-3 (or 1-5) and confidence is 0-1.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build feature vector
        X = self._build_feature_vector(features)

        # Predict
        tier = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = proba[tier - 1]  # Tier is 1-indexed

        return int(tier), float(confidence)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict traffic tiers for multiple domains.

        Args:
            df: DataFrame with GTR features.

        Returns:
            DataFrame with added 'predicted_tier' and 'prediction_confidence' columns.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = self._prepare_features(df)
        tiers = self.model.predict(X)
        probas = self.model.predict_proba(X)
        confidences = [probas[i, tier - 1] for i, tier in enumerate(tiers)]

        result = df.copy()
        result["predicted_tier"] = tiers
        result["prediction_confidence"] = confidences
        return result

    def get_tier_description(self, tier: int) -> str:
        """Get human-readable description for a tier."""
        return self.tier_descriptions.get(tier, f"Unknown Tier {tier}")

    def _build_feature_vector(self, features: dict) -> np.ndarray:
        """Build feature vector from dictionary."""
        values = []
        for feat in self.feature_names:
            val = features.get(feat, 0)
            if val is None:
                val = 0
            values.append(val)

        X = np.array(values).reshape(1, -1)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from DataFrame."""
        # Ensure all feature columns exist
        X = pd.DataFrame()
        for feat in self.feature_names:
            if feat in df.columns:
                X[feat] = df[feat]
            else:
                X[feat] = 0

        # Handle missing values
        X = X.fillna(0)

        # Scale
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X


def load_training_data() -> pd.DataFrame:
    """Load time-aligned GTR training data."""
    path = DATA_DIR / "gtr_time_aligned.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Run scripts/preprocess_gtr_time_aligned.py first."
        )
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for training.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is tier labels.
    """
    X = df[GTR_FEATURES].copy()

    # Handle missing values
    X = X.fillna(0)

    # Log transform heavy-tailed features
    for feat in LOG_TRANSFORM_FEATURES:
        if feat in X.columns:
            X[feat] = np.log1p(X[feat])

    y = df["tier"].values

    return X.values, y


def calculate_adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy within 1 tier."""
    return np.mean(np.abs(y_true - y_pred) <= 1)


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "xgboost",
) -> tuple[object, StandardScaler, ModelMetrics]:
    """
    Train and evaluate a classifier.

    Args:
        X_train, X_test: Feature matrices.
        y_train, y_test: Tier labels.
        model_name: One of 'logreg', 'rf', 'xgboost'.

    Returns:
        Tuple of (model, scaler, metrics).
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize model
    if model_name == "logreg":
        model = LogisticRegression(
                        max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    metrics = ModelMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        f1_macro=f1_score(y_test, y_pred, average="macro"),
        f1_weighted=f1_score(y_test, y_pred, average="weighted"),
        adjacent_accuracy=calculate_adjacent_accuracy(y_test, y_pred),
        confusion_matrix=confusion_matrix(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5"]
        ),
    )

    return model, scaler, metrics


def cross_validate_model(
    X: np.ndarray, y: np.ndarray, model_name: str = "xgboost", n_folds: int = 5
) -> dict:
    """
    Perform stratified k-fold cross-validation.

    Returns:
        Dictionary with CV scores.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_name == "logreg":
        model = LogisticRegression(
                        max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracy_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_macro")

    return {
        "accuracy_mean": accuracy_scores.mean(),
        "accuracy_std": accuracy_scores.std(),
        "f1_macro_mean": f1_scores.mean(),
        "f1_macro_std": f1_scores.std(),
    }
