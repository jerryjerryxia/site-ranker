"""Data schemas for raw signals and feature vectors."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class TakedownStats:
    """Google Transparency Report takedown statistics."""

    total_urls_removed: int
    total_requests: int
    total_urls_no_action: int
    first_seen: Optional[date] = None
    last_seen: Optional[date] = None


@dataclass
class TelegramChannel:
    """Telegram channel information."""

    channel_url: str
    name: str
    subscribers: Optional[int] = None
    description: Optional[str] = None


@dataclass
class RedditSignals:
    """Reddit mention signals."""

    mention_count: int = 0
    upvote_sum: int = 0
    unique_threads: int = 0
    total_comments: int = 0


@dataclass
class SiteMetadata:
    """Metadata scraped from site."""

    telegram_links: list[str] = field(default_factory=list)
    detected_language: Optional[str] = None
    is_arabic: bool = False
    uses_cloudflare: bool = False


@dataclass
class RawSignals:
    """
    Raw signals collected from all data sources.

    This is the output of the collection phase, before feature engineering.
    """

    domain: str

    # Traffic rank signals
    tranco_rank: Optional[int] = None
    tranco_subdomain_rank: Optional[int] = None
    umbrella_rank: Optional[int] = None
    majestic_rank: Optional[int] = None

    # Engagement signals
    telegram_subscribers: Optional[int] = None
    telegram_channel_url: Optional[str] = None
    telegram_channel_name: Optional[str] = None
    reddit_mention_count: int = 0
    reddit_upvote_sum: int = 0
    reddit_unique_threads: int = 0

    # Google Transparency signals
    google_takedown_count: int = 0
    google_urls_removed: int = 0
    google_urls_no_action: int = 0
    google_first_seen: Optional[date] = None
    google_last_seen: Optional[date] = None

    # Major organization enforcement
    major_org_requests: int = 0  # Requests from major anti-piracy orgs
    unique_major_orgs: int = 0  # Count of unique major orgs targeting this domain
    major_org_ratio: float = 0.0  # Ratio of major org requests to total requests
    major_org_requests_last_90d: int = 0  # Recent major org activity (last 90 days)

    # Major studio enforcement
    major_studio_requests: int = 0  # Requests from major studios (Disney, Warner, etc.)
    unique_major_studios: int = 0  # Count of unique major studios targeting this domain

    # Infrastructure signals
    domain_age_days: Optional[int] = None
    subdomain_count: int = 0
    uses_cloudflare: bool = False

    # Regional signals
    is_arabic: bool = False
    is_mena_tld: bool = False
    has_telegram_link_onsite: bool = False
    detected_language: Optional[str] = None


@dataclass
class FeatureVector:
    """
    Engineered features for tier classification.

    This is the output of the feature engineering phase, ready for ML model.
    """

    domain: str

    # Traffic signals (inverted ranks - higher is better)
    traffic_rank_score: float = 0.0
    traffic_confidence: int = 0  # Number of rank sources available
    tranco_score: float = 0.0
    has_tranco: int = 0

    # Engagement signals
    telegram_subscribers_log: float = 0.0
    telegram_engagement: float = 0.0  # Exponential decay score
    has_telegram: int = 0

    # Enforcement signals
    google_takedown_count_log: float = 0.0
    google_urls_removed_log: float = 0.0
    gtr_enforcement: float = 0.0  # Exponential decay score
    removal_rate: float = 0.0  # % of requested URLs removed
    has_google_history: int = 0

    # Raw enforcement counts (for tier boosting)
    google_takedown_count: int = 0
    major_org_requests: int = 0
    unique_major_orgs: int = 0
    major_org_requests_last_90d: int = 0  # Recent activity
    major_studio_requests: int = 0
    unique_major_studios: int = 0

    # Site metadata
    uses_cloudflare: int = 0
    is_mena_relevant: int = 0
    has_telegram_link_onsite: int = 0

    # Composite scores
    engagement_score: float = 0.0  # Weighted combo of telegram + gtr
    piracy_score: float = 0.0  # Overall piracy likelihood score


@dataclass
class TierPrediction:
    """
    Tier prediction result.

    3-Tier System (recommended):
    - Tier 1 = High Impact (>1M monthly visits)
    - Tier 2 = Moderate (10K-1M monthly visits)
    - Tier 3 = Low Impact (<10K monthly visits)

    Legacy 5-Tier System:
    - Tier 1 = highest traffic (>10M monthly)
    - Tier 5 = lowest traffic (<10K monthly)
    """

    domain: str
    tier: int  # 1-3 (3-tier) or 1-5 (5-tier)
    confidence: float  # 0.0-1.0
    is_mena: bool
    prediction_method: str  # 'tranco_rank', 'ml_model_3tier', 'piracy_score', etc.

    # Optional: full feature vector for debugging
    features: Optional[FeatureVector] = None


@dataclass
class RankedSite:
    """
    Final output: site with tier assignment and all metadata.
    """

    domain: str
    tier: int
    confidence: float
    is_mena: bool
    prediction_method: str

    # Content classification
    content_type: str = "unknown"  # video_streaming, adult, file_hosting, social, unknown
    content_confidence: float = 0.0  # 0.0-1.0
    is_target: bool = False  # True if video_streaming OR file_hosting (both are targets)

    # Key signals for sorting/filtering
    tranco_rank: Optional[int] = None
    telegram_subscribers: Optional[int] = None
    google_takedowns: int = 0

    # Scores for reference
    traffic_rank_score: float = 0.0
    engagement_score: float = 0.0
    traffic_confidence: int = 0
