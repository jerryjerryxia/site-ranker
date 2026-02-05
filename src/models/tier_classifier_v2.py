"""
Improved tier classification using 40/40/20 scoring model.

This addresses the issues in the original tier_classifier.py:
1. No double-counting of GTR signals
2. Telegram properly weighted as primary impact signal
3. Single coherent scoring path (no separate Tranco vs score paths)
4. Clear separation: Impact (user behavior) vs Activity (enforcement) vs Quality (content value)

Scoring Formula:
- 40% User Impact (Telegram 60% + Tranco 40%)
- 40% Enforcement Activity (Volume 50% + Volume Trend 30% + Breadth 20%)
- 20% Content Quality (Major studio requests 60% + Studio ratio 40%)
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ScoringWeights:
    """Weights for the 40/40/20 scoring model."""

    # Dimension weights (must sum to 1.0)
    impact_weight: float = 0.40
    activity_weight: float = 0.40
    quality_weight: float = 0.20

    # Impact sub-weights (must sum to 1.0)
    telegram_weight: float = 0.60
    tranco_weight: float = 0.40

    # Activity sub-weights (must sum to 1.0)
    volume_weight: float = 0.50
    volume_trend_weight: float = 0.30
    breadth_weight: float = 0.20

    # Quality sub-weights (must sum to 1.0)
    studio_requests_weight: float = 0.60
    studio_ratio_weight: float = 0.40


@dataclass
class TierThresholds:
    """Thresholds for tier assignment based on priority score."""

    tier1_min_score: float = 0.70  # Top tier - very high priority
    tier2_min_score: float = 0.50  # High priority
    tier3_min_score: float = 0.30  # Medium priority
    tier4_min_score: float = 0.15  # Low priority
    # tier5: < 0.15 - very low priority or insufficient data


class TierClassifierV2:
    """
    Improved tier classifier using 40/40/20 scoring model.

    Key improvements:
    - No double-counting of signals
    - Telegram properly valued as primary impact metric
    - Single coherent scoring path regardless of available signals
    - Transparent, explainable calculations
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        thresholds: Optional[TierThresholds] = None,
    ):
        """
        Initialize tier classifier.

        Args:
            weights: Scoring weights (uses defaults if None)
            thresholds: Tier thresholds (uses defaults if None)
        """
        self.weights = weights or ScoringWeights()
        self.thresholds = thresholds or TierThresholds()

    def calculate_priority_score(
        self,
        # Impact signals
        telegram_subscribers: Optional[int] = None,
        tranco_rank: Optional[int] = None,
        total_urls_removed: Optional[int] = None,
        # Activity signals
        major_org_requests_last_90d: Optional[int] = None,
        major_org_requests_last_30d: Optional[int] = None,
        unique_major_orgs: Optional[int] = None,
        avg_urls_per_request: Optional[float] = None,
        # Quality signals
        major_studio_requests: Optional[int] = None,
        major_org_requests: Optional[int] = None,
    ) -> dict:
        """
        Calculate priority score using 40/40/20 model.

        Returns:
            Dict with:
                - priority_score: Final composite score (0.0-1.0)
                - impact_score: User impact dimension (0.0-1.0)
                - activity_score: Enforcement activity dimension (0.0-1.0)
                - quality_score: Content quality dimension (0.0-1.0)
                - tier: Tier classification (1-5)
                - confidence: Confidence in classification (0.0-1.0)
                - signals_available: Count of non-null signals
        """
        # === Impact Score (40%) ===
        impact_score, impact_signals = self._calculate_impact(
            telegram_subscribers, tranco_rank, total_urls_removed
        )

        # === Activity Score (40%) ===
        activity_score, activity_signals = self._calculate_activity(
            major_org_requests_last_90d,
            major_org_requests_last_30d,
            unique_major_orgs,
            avg_urls_per_request,
        )

        # === Quality Score (20%) ===
        quality_score, quality_signals = self._calculate_quality(
            major_studio_requests, major_org_requests
        )

        # === Composite Priority Score ===
        priority_score = (
            self.weights.impact_weight * impact_score +
            self.weights.activity_weight * activity_score +
            self.weights.quality_weight * quality_score
        )

        # === Tier Classification ===
        tier = self._score_to_tier(priority_score)

        # === Confidence Calculation ===
        total_signals = impact_signals + activity_signals + quality_signals
        max_signals = 7  # telegram, tranco, volume(90d+avg_urls), volume_trend(30d+90d+avg_urls), unique_orgs, studio_requests, major_org_requests
        confidence = total_signals / max_signals

        return {
            'priority_score': priority_score,
            'impact_score': impact_score,
            'activity_score': activity_score,
            'quality_score': quality_score,
            'tier': tier,
            'confidence': confidence,
            'signals_available': total_signals,
        }

    def _calculate_impact(
        self,
        telegram_subscribers: Optional[int],
        tranco_rank: Optional[int],
        total_urls_removed: Optional[int] = None,
        avg_urls_per_request: Optional[float] = None,
    ) -> tuple[float, int]:
        """
        Calculate impact score from user engagement signals.

        Impact = 0.60 * telegram_score + 0.40 * tranco_score

        If no Telegram or Tranco data is available, falls back to volume-based proxy:
        - If total_urls_removed >= 116 (P90 for non-Tranco sites with non-zero volume), use 0.5 impact score

        Args:
            telegram_subscribers: Telegram channel subscribers
            tranco_rank: Tranco traffic rank (1 = most popular)
            total_urls_removed: Total URLs removed (for fallback)
            avg_urls_per_request: Average URLs per request (unused, kept for compatibility)

        Returns:
            (impact_score, signals_count)
        """
        signals = 0
        telegram_score = 0.0
        tranco_score = 0.0

        # Telegram score (exponential decay with half-life at 100K)
        if telegram_subscribers and telegram_subscribers > 0:
            telegram_score = self._exponential_decay(telegram_subscribers, half_life=100_000)
            signals += 1

        # Tranco score (inverted rank with max rank 10M)
        if tranco_rank and tranco_rank > 0:
            tranco_score = self._invert_rank(tranco_rank, max_rank=10_000_000)
            signals += 1

        # Volume-based fallback for sites without Tranco/Telegram
        # If a site has high total removal volume, it's likely to have significant impact
        if signals == 0:
            if total_urls_removed and total_urls_removed >= 116:
                # Threshold at P90 (116 URLs) for non-Tranco sites with non-zero volume
                return 0.5, 1  # Medium impact based on volume proxy
            return 0.0, 0

        impact = 0.0
        if telegram_subscribers:
            impact += self.weights.telegram_weight * telegram_score
        if tranco_rank:
            impact += self.weights.tranco_weight * tranco_score

        # Normalize if only one signal available
        if signals == 1:
            if telegram_subscribers:
                impact = telegram_score  # Use telegram score directly
            else:
                impact = tranco_score  # Use tranco score directly

        return impact, signals

    def _calculate_activity(
        self,
        major_org_requests_last_90d: Optional[int],
        major_org_requests_last_30d: Optional[int],
        unique_major_orgs: Optional[int],
        avg_urls_per_request: Optional[float] = None,
    ) -> tuple[float, int]:
        """
        Calculate activity score from enforcement signals.

        Activity = 0.50 * volume_score + 0.30 * volume_trend_score + 0.20 * breadth_score

        Args:
            major_org_requests_last_90d: Requests in last 90 days
            major_org_requests_last_30d: Requests in last 30 days
            unique_major_orgs: Number of unique major orgs
            avg_urls_per_request: Average URLs per request (used to estimate volume)

        Returns:
            (activity_score, signals_count)
        """
        signals = 0

        # Volume score (recent 90d URL removal volume)
        # Estimate: avg_urls_per_request × major_org_requests_last_90d
        volume_score = 0.0
        if (major_org_requests_last_90d and major_org_requests_last_90d > 0 and
            avg_urls_per_request and avg_urls_per_request > 0):
            estimated_url_volume_90d = avg_urls_per_request * major_org_requests_last_90d
            # Use exponential decay with half-life at 100K URLs
            volume_score = self._exponential_decay(estimated_url_volume_90d, half_life=100_000)
            signals += 1

        # Volume trend score (acceleration in URL removal volume)
        volume_trend_score = 0.0
        if (major_org_requests_last_30d is not None and
            major_org_requests_last_90d and major_org_requests_last_90d > 0 and
            avg_urls_per_request and avg_urls_per_request > 0):

            # Estimate volume for 90d and 30d periods
            estimated_url_volume_90d = avg_urls_per_request * major_org_requests_last_90d
            estimated_url_volume_30d = avg_urls_per_request * major_org_requests_last_30d

            # Calculate monthly volume rates
            vol_rate_90d = estimated_url_volume_90d / 3.0  # Monthly rate over 90 days
            vol_rate_30d = estimated_url_volume_30d  # Monthly rate for last 30 days

            if vol_rate_90d > 0:
                # Volume acceleration ratio: > 1.0 = ramping up, < 1.0 = winding down
                vol_accel_ratio = vol_rate_30d / vol_rate_90d
                # Map to 0-1 score: 2x = 1.0, 1x = 0.5, 0x = 0.0
                volume_trend_score = min(1.0, vol_accel_ratio / 2.0)
                signals += 1

        # Breadth score (variety of targeting orgs, exponential decay with half-life at 5)
        breadth_score = 0.0
        if unique_major_orgs and unique_major_orgs > 0:
            breadth_score = self._exponential_decay(unique_major_orgs, half_life=5)
            signals += 1

        if signals == 0:
            return 0.0, 0

        # Calculate weighted activity score
        activity = (
            self.weights.volume_weight * volume_score +
            self.weights.volume_trend_weight * volume_trend_score +
            self.weights.breadth_weight * breadth_score
        )

        return activity, signals

    def _calculate_quality(
        self,
        major_studio_requests: Optional[int],
        major_org_requests: Optional[int],
    ) -> tuple[float, int]:
        """
        Calculate quality score from content value signals.

        Quality = 0.60 * studio_requests_score + 0.40 * studio_ratio

        Args:
            major_studio_requests: Requests from major studios
            major_org_requests: Total requests from major orgs

        Returns:
            (quality_score, signals_count)
        """
        signals = 0

        # Studio requests score (exponential decay with half-life at 10K)
        studio_score = 0.0
        if major_studio_requests and major_studio_requests > 0:
            studio_score = self._exponential_decay(major_studio_requests, half_life=10_000)
            signals += 1

        # Studio ratio (premium content ratio)
        ratio_score = 0.0
        if (major_studio_requests is not None and
            major_org_requests and major_org_requests > 0):
            ratio = major_studio_requests / major_org_requests
            ratio_score = min(1.0, ratio)  # Cap at 1.0
            signals += 1

        if signals == 0:
            return 0.0, 0

        # Calculate weighted quality score
        quality = (
            self.weights.studio_requests_weight * studio_score +
            self.weights.studio_ratio_weight * ratio_score
        )

        return quality, signals

    def _score_to_tier(self, priority_score: float) -> int:
        """
        Convert priority score to tier (1-5).

        Args:
            priority_score: Composite priority score (0.0-1.0)

        Returns:
            Tier classification (1 = highest, 5 = lowest)
        """
        if priority_score >= self.thresholds.tier1_min_score:
            return 1
        elif priority_score >= self.thresholds.tier2_min_score:
            return 2
        elif priority_score >= self.thresholds.tier3_min_score:
            return 3
        elif priority_score >= self.thresholds.tier4_min_score:
            return 4
        else:
            return 5

    def _exponential_decay(self, value: float, half_life: float) -> float:
        """
        Calculate exponential decay score.

        Score approaches 1.0 as value increases.
        At half_life, score = 0.5.

        Formula: 1 - (0.5 ^ (value / half_life))

        Args:
            value: Input value
            half_life: Value at which score = 0.5

        Returns:
            Score between 0.0 and 1.0
        """
        if value <= 0:
            return 0.0
        return 1.0 - math.pow(0.5, value / half_life)

    def _invert_rank(self, rank: int, max_rank: int = 10_000_000) -> float:
        """
        Invert rank to score (lower rank = higher score).

        Formula: 1.0 - (rank - 1) / max_rank

        Args:
            rank: Rank (1 = best)
            max_rank: Maximum rank to consider

        Returns:
            Score between 0.0 and 1.0
        """
        if rank <= 0:
            return 0.0
        if rank >= max_rank:
            return 0.0
        return 1.0 - (rank - 1) / max_rank
