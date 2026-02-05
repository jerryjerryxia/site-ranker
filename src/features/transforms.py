"""
Feature transformation utilities.

Provides common transformations for converting raw signals into
normalized, comparable features for ML models.
"""

import math
from typing import List, Optional


def log1p(x: Optional[float]) -> float:
    """
    Log transform with +1 offset for heavy-tailed distributions.

    Useful for count data (subscribers, mentions, takedowns) that
    span multiple orders of magnitude.

    Args:
        x: Value to transform (can be None)

    Returns:
        log(1 + x) if x is not None, else 0.0

    Examples:
        >>> log1p(0)
        0.0
        >>> log1p(99)  # log(100) ≈ 4.6
        4.605170185988092
        >>> log1p(None)
        0.0
    """
    if x is None:
        return 0.0
    return math.log1p(max(0, x))


def invert_rank(rank: Optional[int], max_rank: int = 10_000_000) -> float:
    """
    Convert rank to score (higher = better).

    Lower ranks (e.g., #1) should produce higher scores.
    Missing ranks are treated as worse than max_rank.

    Args:
        rank: Tranco/Umbrella/Majestic rank (1-indexed, lower is better)
        max_rank: Maximum rank to consider

    Returns:
        Normalized score from 0.0 (worst) to 1.0 (best)

    Examples:
        >>> invert_rank(1, max_rank=1_000_000)
        1.0
        >>> invert_rank(500_000, max_rank=1_000_000)
        0.5
        >>> invert_rank(1_000_000, max_rank=1_000_000)
        0.0
        >>> invert_rank(None)
        0.0
    """
    if rank is None:
        return 0.0

    # Clamp rank to max_rank
    rank = min(rank, max_rank)

    # Invert: rank 1 → 1.0, rank max_rank → 0.0
    return 1.0 - (rank - 1) / max_rank


def bucket_age(days: Optional[int]) -> int:
    """
    Bin domain age into categorical buckets.

    Buckets:
    - 0: Unknown/missing
    - 1: Very new (0-180 days / ~6 months)
    - 2: New (180-730 days / 6 months - 2 years)
    - 3: Established (730-1825 days / 2-5 years)
    - 4: Old (1825+ days / 5+ years)

    Args:
        days: Domain age in days

    Returns:
        Bucket 0-4

    Examples:
        >>> bucket_age(None)
        0
        >>> bucket_age(100)  # 3 months
        1
        >>> bucket_age(365)  # 1 year
        2
        >>> bucket_age(1000)  # ~3 years
        3
        >>> bucket_age(3000)  # 8 years
        4
    """
    if days is None:
        return 0

    if days < 180:
        return 1
    elif days < 730:
        return 2
    elif days < 1825:
        return 3
    else:
        return 4


def normalize_percentile(values: List[float], value: float) -> float:
    """
    Normalize value to percentile within a list of values.

    Args:
        values: List of all values (for context)
        value: Value to normalize

    Returns:
        Percentile rank from 0.0 to 1.0

    Examples:
        >>> values = [1, 2, 3, 4, 5]
        >>> normalize_percentile(values, 1)
        0.0
        >>> normalize_percentile(values, 3)
        0.5
        >>> normalize_percentile(values, 5)
        1.0
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)

    # Count how many values are less than or equal to target
    count_below = sum(1 for v in sorted_values if v <= value)

    # Return percentile
    return count_below / len(sorted_values)


def clip(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clip value to range [min_val, max_val].

    Args:
        value: Value to clip
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped value

    Examples:
        >>> clip(0.5)
        0.5
        >>> clip(-0.1)
        0.0
        >>> clip(1.5)
        1.0
        >>> clip(2.5, min_val=0.0, max_val=5.0)
        2.5
    """
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide with safe handling of zero denominator.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero

    Returns:
        numerator / denominator, or default if denominator is 0

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=1.0)
        1.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def exponential_decay_score(value: float, half_life: float = 100_000) -> float:
    """
    Convert value to score using exponential decay.

    Useful when higher values are better, but with diminishing returns.

    Args:
        value: Input value
        half_life: Value at which score reaches 0.5

    Returns:
        Score from 0.0 to 1.0

    Examples:
        >>> exponential_decay_score(0)
        0.0
        >>> exponential_decay_score(100_000)  # At half-life
        0.5
        >>> exponential_decay_score(1_000_000)
        0.9990234375
    """
    if value <= 0:
        return 0.0

    # 1 - exp(-value * ln(2) / half_life)
    # At value=half_life, this equals 0.5
    decay_rate = math.log(2) / half_life
    return 1.0 - math.exp(-value * decay_rate)
