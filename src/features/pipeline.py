"""
Feature engineering pipeline.

Transforms raw signals into engineered features for tier classification.
"""

from dataclasses import dataclass
from typing import Optional

from src.features.schema import FeatureVector, RawSignals
from src.features.transforms import (
    exponential_decay_score,
    invert_rank,
    log1p,
    safe_divide,
)


class FeaturePipeline:
    """
    Transforms RawSignals into FeatureVector.

    Handles:
    - Missing value imputation
    - Log transforms for heavy-tailed distributions
    - Rank inversion (lower rank → higher score)
    - Composite score computation
    """

    def __init__(
        self,
        max_rank: int = 10_000_000,
        telegram_half_life: float = 100_000,
        takedown_half_life: float = 10_000,
    ):
        """
        Initialize feature pipeline.

        Args:
            max_rank: Maximum rank for traffic lists (default 10M)
            telegram_half_life: Subscribers count for 0.5 engagement score
            takedown_half_life: Takedown count for 0.5 enforcement score
        """
        self.max_rank = max_rank
        self.telegram_half_life = telegram_half_life
        self.takedown_half_life = takedown_half_life

    def transform(self, raw: RawSignals) -> FeatureVector:
        """
        Transform raw signals into feature vector.

        Args:
            raw: Raw signals from collectors

        Returns:
            Engineered feature vector
        """
        # === Traffic Signals ===

        # Tranco
        tranco_score = invert_rank(raw.tranco_rank, self.max_rank)
        has_tranco = 1 if raw.tranco_rank is not None else 0

        # Tranco subdomain (fallback)
        tranco_sub_score = invert_rank(raw.tranco_subdomain_rank, self.max_rank)
        has_tranco_sub = 1 if raw.tranco_subdomain_rank is not None else 0

        # Use subdomain rank as fallback if apex not found
        if has_tranco == 0 and has_tranco_sub == 1:
            tranco_score = tranco_sub_score
            has_tranco = 1

        # Composite traffic score (for now, just Tranco)
        # Later: add Umbrella, Majestic with weights
        traffic_rank_score = tranco_score
        traffic_confidence = has_tranco

        # === Engagement Signals ===

        # Telegram
        telegram_subscribers_log = log1p(raw.telegram_subscribers)
        has_telegram = 1 if raw.telegram_subscribers is not None and raw.telegram_subscribers > 0 else 0

        # Telegram engagement score (exponential decay)
        telegram_engagement = exponential_decay_score(
            raw.telegram_subscribers or 0,
            self.telegram_half_life
        )

        # === Enforcement Signals ===

        # Google Transparency Report
        google_takedown_count_log = log1p(raw.google_takedown_count)
        google_urls_removed_log = log1p(raw.google_urls_removed)
        has_google_history = 1 if raw.google_takedown_count and raw.google_takedown_count > 0 else 0

        # GTR enforcement score (exponential decay on takedown count)
        gtr_enforcement = exponential_decay_score(
            raw.google_takedown_count or 0,
            self.takedown_half_life
        )

        # Removal rate (what % of requested URLs were actually removed)
        total_urls = (raw.google_urls_removed or 0) + (raw.google_urls_no_action or 0)
        removal_rate = safe_divide(raw.google_urls_removed or 0, total_urls, default=0.0)

        # === Site Metadata ===

        uses_cloudflare = 1 if raw.uses_cloudflare else 0
        is_arabic = 1 if raw.is_arabic else 0
        is_mena_tld = 1 if raw.is_mena_tld else 0
        has_telegram_link_onsite = 1 if raw.has_telegram_link_onsite else 0

        # MENA relevance (either Arabic content OR MENA TLD)
        is_mena_relevant = 1 if (is_arabic or is_mena_tld) else 0

        # === Composite Scores ===

        # Engagement score: weighted combination of signals
        # - Telegram is primary engagement signal (weight 0.7)
        # - GTR takedowns indicate notoriety/popularity (weight 0.3)
        engagement_score = (
            0.7 * telegram_engagement +
            0.3 * gtr_enforcement
        )

        # Overall piracy score (combines traffic, engagement, enforcement)
        # Higher = more likely to be high-tier piracy site
        piracy_score = (
            0.4 * traffic_rank_score +
            0.4 * engagement_score +
            0.2 * gtr_enforcement
        )

        # === Create Feature Vector ===

        return FeatureVector(
            domain=raw.domain,
            # Traffic signals
            traffic_rank_score=traffic_rank_score,
            traffic_confidence=traffic_confidence,
            tranco_score=tranco_score,
            has_tranco=has_tranco,
            # Engagement signals
            telegram_subscribers_log=telegram_subscribers_log,
            telegram_engagement=telegram_engagement,
            has_telegram=has_telegram,
            # Enforcement signals
            google_takedown_count_log=google_takedown_count_log,
            google_urls_removed_log=google_urls_removed_log,
            gtr_enforcement=gtr_enforcement,
            removal_rate=removal_rate,
            has_google_history=has_google_history,
            google_takedown_count=raw.google_takedown_count,
            major_org_requests=raw.major_org_requests,
            unique_major_orgs=raw.unique_major_orgs,
            major_org_requests_last_90d=raw.major_org_requests_last_90d,
            major_studio_requests=raw.major_studio_requests,
            unique_major_studios=raw.unique_major_studios,
            # Site metadata
            uses_cloudflare=uses_cloudflare,
            is_mena_relevant=is_mena_relevant,
            has_telegram_link_onsite=has_telegram_link_onsite,
            # Composite scores
            engagement_score=engagement_score,
            piracy_score=piracy_score,
        )

    def transform_batch(self, raw_signals: list[RawSignals]) -> list[FeatureVector]:
        """
        Transform a batch of raw signals.

        Args:
            raw_signals: List of raw signals

        Returns:
            List of feature vectors
        """
        return [self.transform(raw) for raw in raw_signals]
