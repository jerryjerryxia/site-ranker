"""
Tier classification for piracy sites.

3-Tier System (recommended):
- Tier 1 = High Impact (>1M monthly visits)
- Tier 2 = Moderate (10K-1M monthly visits)
- Tier 3 = Low Impact (<10K monthly visits)

Legacy 5-Tier System:
- Tier 1 = Highest traffic (top 10K globally)
- Tier 2 = High traffic (10K-100K)
- Tier 3 = Medium traffic (100K-500K)
- Tier 4 = Low traffic (500K-1M)
- Tier 5 = Very low traffic or unknown
"""

from dataclasses import dataclass
from typing import Optional

from src.features.schema import FeatureVector, TierPrediction


@dataclass
class TierThresholds:
    """Thresholds for tier assignment based on Tranco rank."""

    # 3-tier system (based on traffic volume, not global rank)
    # Maps Tranco rank to approximate monthly visits:
    # - Rank ~50K = ~1M visits (Tier 1 threshold)
    # - Rank ~500K = ~10K visits (Tier 2 threshold)
    tier1_max_rank: int = 50_000  # >1M visits
    tier2_max_rank: int = 500_000  # 10K-1M visits
    # Tier 3 = everything else (<10K visits)

    # For sites without Tranco rank, use piracy_score
    tier1_min_score: float = 0.7
    tier2_min_score: float = 0.4

    # Number of tiers (3 or 5)
    num_tiers: int = 3


class TierClassifier:
    """
    Rule-based tier classifier.

    Primary strategy: Use Tranco rank for direct assignment
    Fallback strategy: Use composite piracy_score for unranked sites
    """

    def __init__(self, thresholds: Optional[TierThresholds] = None):
        """
        Initialize tier classifier.

        Args:
            thresholds: Custom tier thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or TierThresholds()

    def predict(self, features: FeatureVector) -> TierPrediction:
        """
        Predict tier for a site.

        Args:
            features: Engineered feature vector

        Returns:
            Tier prediction with confidence score
        """
        # Strategy 1: Direct assignment based on Tranco rank
        if features.has_tranco:
            prediction = self._predict_from_rank(features)
        else:
            # Strategy 2: Use composite piracy score for unranked sites
            prediction = self._predict_from_score(features)

        # Strategy 3: GTR-based tier boosting (overrides if enforcement is very high)
        prediction = self._apply_gtr_boost(prediction, features)

        return prediction

    def _predict_from_rank(self, features: FeatureVector) -> TierPrediction:
        """
        Predict tier directly from Tranco rank.

        This is high-confidence since Tranco is a proven traffic proxy.

        3-Tier Mapping (based on approximate traffic):
        - Rank <= 50K → Tier 1 (High Impact, >1M visits)
        - Rank 50K-500K → Tier 2 (Moderate, 10K-1M visits)
        - Rank > 500K → Tier 3 (Low Impact, <10K visits)
        """
        # Extract raw rank from tranco_score (reverse the inversion)
        # tranco_score = 1.0 - (rank - 1) / max_rank
        # rank = (1.0 - tranco_score) * max_rank + 1
        max_rank = 10_000_000
        estimated_rank = int((1.0 - features.tranco_score) * max_rank + 1)

        # Apply 3-tier thresholds
        if estimated_rank <= self.thresholds.tier1_max_rank:
            tier = 1  # High Impact
        elif estimated_rank <= self.thresholds.tier2_max_rank:
            tier = 2  # Moderate
        else:
            tier = 3  # Low Impact

        return TierPrediction(
            domain=features.domain,
            tier=tier,
            confidence=1.0,  # High confidence - direct from rank
            is_mena=bool(features.is_mena_relevant),
            prediction_method="tranco_rank",
            features=features,
        )

    def _predict_from_score(self, features: FeatureVector) -> TierPrediction:
        """
        Predict tier from composite piracy score.

        Lower confidence since we're estimating without traffic data.

        3-Tier Score Thresholds:
        - Score >= 0.7 → Tier 1 (High Impact)
        - Score >= 0.4 → Tier 2 (Moderate)
        - Score < 0.4 → Tier 3 (Low Impact)
        """
        score = features.piracy_score

        # Apply 3-tier score thresholds
        if score >= self.thresholds.tier1_min_score:
            tier = 1  # High Impact
        elif score >= self.thresholds.tier2_min_score:
            tier = 2  # Moderate
        else:
            tier = 3  # Low Impact

        # Confidence based on available signals
        confidence = self._calculate_confidence(features)

        # Adjust tier down if confidence is very low
        if confidence < 0.3 and tier < 3:
            tier += 1  # Be conservative with low-confidence predictions

        return TierPrediction(
            domain=features.domain,
            tier=tier,
            confidence=confidence,
            is_mena=bool(features.is_mena_relevant),
            prediction_method="piracy_score",
            features=features,
        )

    def _apply_gtr_boost(self, prediction: TierPrediction, features: FeatureVector) -> TierPrediction:
        """
        Apply GTR-based tier boosting for high-enforcement sites.

        If a site has very high GTR takedown counts (especially from major orgs),
        it's a strong signal of high-impact piracy even without Tranco rank.

        3-Tier Boost Thresholds:
        - Tier 1 (High Impact): 5K+ recent major org activity OR 50K+ studio OR 100K+ major org
        - Tier 2 (Moderate): 2K+ recent OR 10K+ studio OR 25K+ major org OR 50K+ total
        - No boost to Tier 3 (sites without significant GTR stay low impact)

        Boost priority (highest to lowest):
        1. Recent major org activity (last 90 days) - indicates current relevance
        2. Major studio enforcement - high-value copyright holders
        3. Historical major org requests - proven high-impact sites
        4. Total GTR requests - fallback for unranked sites with enforcement
        """
        # Check if we have GTR data in features
        google_takedowns = getattr(features, 'google_takedown_count', 0) or 0
        major_org_requests = getattr(features, 'major_org_requests', 0) or 0
        major_org_requests_last_90d = getattr(features, 'major_org_requests_last_90d', 0) or 0
        major_studio_requests = getattr(features, 'major_studio_requests', 0) or 0

        if google_takedowns == 0 and major_org_requests == 0:
            return prediction  # No GTR data, no boost

        original_tier = prediction.tier
        boosted_tier = original_tier
        boost_reason = None

        # PRIORITY 1: Recent major org activity (last 90 days)
        # This is the strongest signal - indicates currently active high-impact piracy
        if major_org_requests_last_90d >= 5_000:
            boosted_tier = min(boosted_tier, 1)  # High Impact
            boost_reason = "recent_major_org_5k+"
        elif major_org_requests_last_90d >= 2_000:
            boosted_tier = min(boosted_tier, 2)  # Moderate
            boost_reason = "recent_major_org_2k+"

        # PRIORITY 2: Major studio enforcement (if no recent activity boost)
        # High-value copyright holders targeting the site
        if boost_reason is None:
            if major_studio_requests >= 50_000:
                boosted_tier = min(boosted_tier, 1)  # High Impact
                boost_reason = "major_studio_50k+"
            elif major_studio_requests >= 10_000:
                boosted_tier = min(boosted_tier, 2)  # Moderate
                boost_reason = "major_studio_10k+"

        # PRIORITY 3: Historical major org requests (if no studio boost)
        if boost_reason is None:
            if major_org_requests >= 100_000:
                boosted_tier = min(boosted_tier, 1)  # High Impact
                boost_reason = "major_org_100k+"
            elif major_org_requests >= 25_000:
                boosted_tier = min(boosted_tier, 2)  # Moderate
                boost_reason = "major_org_25k+"

        # PRIORITY 4: Total request boosting (fallback if no major org/studio boost)
        if boost_reason is None:
            if google_takedowns >= 50_000:
                boosted_tier = min(boosted_tier, 2)  # Moderate
                boost_reason = "gtr_50k+"

        # If boosted, update prediction
        if boosted_tier < original_tier:
            return TierPrediction(
                domain=prediction.domain,
                tier=boosted_tier,
                confidence=max(prediction.confidence, 0.8),  # High confidence for GTR boost
                is_mena=prediction.is_mena,
                prediction_method=f"{prediction.prediction_method}+gtr_boost({boost_reason})",
                features=features,
            )

        return prediction

    def _calculate_confidence(self, features: FeatureVector) -> float:
        """
        Calculate prediction confidence based on available signals.

        More signals → higher confidence
        """
        signal_count = 0
        signal_weights = []

        # Telegram signal (weight 0.4)
        if features.has_telegram:
            signal_count += 1
            signal_weights.append(0.4)

        # GTR signal (weight 0.3)
        if features.has_google_history:
            signal_count += 1
            signal_weights.append(0.3)

        # Site metadata (weight 0.2)
        if features.has_telegram_link_onsite:
            signal_count += 1
            signal_weights.append(0.2)

        # Cloudflare (weight 0.1)
        if features.uses_cloudflare:
            signal_count += 1
            signal_weights.append(0.1)

        # No signals → very low confidence
        if signal_count == 0:
            return 0.1

        # Confidence = sum of available signal weights
        confidence = sum(signal_weights)

        return min(1.0, confidence)

    def predict_batch(self, features_list: list[FeatureVector]) -> list[TierPrediction]:
        """
        Predict tiers for a batch of sites.

        Args:
            features_list: List of feature vectors

        Returns:
            List of tier predictions
        """
        return [self.predict(features) for features in features_list]


class HybridClassifier(TierClassifier):
    """
    Hybrid classifier that combines rule-based and ML approaches.

    Uses 3-tier system:
    - Tier 1: High Impact (>1M monthly visits)
    - Tier 2: Moderate (10K-1M monthly visits)
    - Tier 3: Low Impact (<10K monthly visits)

    Priority:
    1. Tranco rank for sites with direct traffic measurements (high confidence)
    2. ML model for sites without Tranco rank (trained on GTR features)
    3. GTR-based tier boosting as a safety net
    """

    def __init__(
        self,
        thresholds: Optional[TierThresholds] = None,
        use_ml: bool = True,
        use_3tier: bool = True,
    ):
        """
        Initialize hybrid classifier.

        Args:
            thresholds: Tier thresholds for rule-based approach
            use_ml: Whether to use ML model for unranked sites
            use_3tier: Whether to use 3-tier system (default) or 5-tier
        """
        super().__init__(thresholds)
        self.use_ml = use_ml
        self.use_3tier = use_3tier
        self.ml_predictor = None

        if use_ml:
            self._load_ml_model()

    def _load_ml_model(self) -> bool:
        """Load ML model for tier prediction."""
        try:
            from src.models.traffic_predictor import TrafficTierPredictor

            self.ml_predictor = TrafficTierPredictor(use_3tier=self.use_3tier)
            if self.ml_predictor.load():
                return True
            else:
                self.ml_predictor = None
                return False
        except Exception:
            self.ml_predictor = None
            return False

    def _predict_from_score(self, features: FeatureVector) -> TierPrediction:
        """
        Predict tier using ML model when available, otherwise use rule-based.

        The 3-tier ML model was trained on 68,851 domains with SimilarWeb ground truth
        and achieves:
        - 80.4% accuracy
        - 60.5% F1 macro
        - 47.9% Tier 1 recall (catches ~half of high-impact sites)
        """
        # Try ML prediction if model is available
        if self.ml_predictor is not None:
            try:
                # Build feature dict from FeatureVector
                # Note: Some features may not be available in FeatureVector
                # The model will handle missing values (defaults to 0)
                gtr_features = {
                    "total_requests": getattr(features, "google_takedown_count", 0) or 0,
                    "total_urls_removed": 0,
                    "total_urls_not_indexed": 0,
                    "total_urls_no_action": 0,
                    "total_urls_targeted": 0,
                    "removal_rate": getattr(features, "removal_rate", 0) or 0,
                    "avg_urls_per_request": 0,
                    "enforcement_duration_days": 0,
                    "days_since_last_request": 0,
                    "requests_per_month": 0,
                    "major_org_requests": getattr(features, "major_org_requests", 0) or 0,
                    "unique_major_orgs": getattr(features, "unique_major_orgs", 0) or 0,
                    "major_org_ratio": 0,
                    "major_studio_requests": getattr(features, "major_studio_requests", 0) or 0,
                    "unique_major_studios": getattr(features, "unique_major_studios", 0) or 0,
                    "major_studio_ratio": 0,
                    "requests_last_30d": 0,
                    "requests_last_90d": getattr(features, "major_org_requests_last_90d", 0) or 0,
                    "major_org_requests_last_30d": 0,
                    "major_org_requests_last_90d": getattr(features, "major_org_requests_last_90d", 0) or 0,
                }

                tier, confidence = self.ml_predictor.predict(gtr_features)

                return TierPrediction(
                    domain=features.domain,
                    tier=tier,
                    confidence=confidence,
                    is_mena=bool(features.is_mena_relevant),
                    prediction_method="ml_model_3tier" if self.use_3tier else "ml_model",
                    features=features,
                )
            except Exception:
                # Fall back to rule-based if ML fails
                pass

        # Fallback to rule-based scoring
        return super()._predict_from_score(features)
