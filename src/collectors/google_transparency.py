"""Google Transparency Report collector.

Uses preprocessed parquet file to provide copyright takedown statistics.
"""

from pathlib import Path
from typing import Optional

import polars as pl

from src.collectors.base import CollectorResult, LocalCollector
from src.features.schema import TakedownStats
from src.utils.domain_utils import normalize_domain


class GoogleTransparencyCollector(LocalCollector):
    """
    Collector for Google Transparency Report copyright data.

    Uses preprocessed parquet file with aggregated domain statistics.
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize Google Transparency collector.

        Args:
            data_path: Path to preprocessed parquet file
        """
        if data_path is None:
            project_root = Path(__file__).parent.parent.parent
            # Use enhanced file with studio and recency data
            data_path = project_root / "data" / "processed" / "google_transparency_with_orgs.parquet"

        super().__init__(name="GoogleTransparency", data_path=str(data_path))

        # Will hold the full dataframe in memory (only ~131 MB)
        self.df: Optional[pl.DataFrame] = None

    def load_data(self) -> None:
        """Load preprocessed parquet file into memory."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(
                f"Preprocessed Google Transparency data not found: {self.data_path}\n"
                "Run scripts/preprocess_google_transparency_chunked.py first."
            )

        self.df = pl.read_parquet(self.data_path)
        self._loaded = True

    def _lookup(self, normalized_domain: str) -> CollectorResult:
        """
        Look up domain in Google Transparency data.

        Args:
            normalized_domain: Normalized domain string

        Returns:
            CollectorResult with takedown statistics
        """
        if self.df is None:
            return CollectorResult(
                domain=normalized_domain,
                success=False,
                error="Data not loaded"
            )

        # Query the dataframe
        result = self.df.filter(pl.col("Domain") == normalized_domain)

        if len(result) == 0:
            # Domain not found (no takedown requests)
            return CollectorResult(
                domain=normalized_domain,
                success=True,  # Success but zero requests
                data={
                    "total_requests": 0,
                    "total_urls_removed": 0,
                    "total_urls_no_action": 0,
                    "total_urls_not_indexed": 0,
                    "has_takedown_history": False,
                }
            )

        # Extract row data
        row = result.row(0, named=True)

        # Create TakedownStats object
        stats = TakedownStats(
            total_urls_removed=row["total_urls_removed"],
            total_requests=row["total_requests"],
            total_urls_no_action=row["total_urls_no_action"],
        )

        # Extract major org data if available (may not exist in older parquet files)
        major_org_requests = row.get("major_org_requests", 0)
        unique_major_orgs = row.get("unique_major_orgs", 0)
        major_org_ratio = row.get("major_org_ratio", 0.0)
        major_org_requests_last_90d = row.get("major_org_requests_last_90d", 0)

        # Extract major studio data
        major_studio_requests = row.get("major_studio_requests", 0)
        unique_major_studios = row.get("unique_major_studios", 0)

        return CollectorResult(
            domain=normalized_domain,
            success=True,
            data={
                "total_requests": row["total_requests"],
                "total_urls_removed": row["total_urls_removed"],
                "total_urls_no_action": row["total_urls_no_action"],
                "total_urls_not_indexed": row["total_urls_not_indexed"],
                "total_urls_pending": row["total_urls_pending"],
                "has_abuser_flag": row["has_abuser_flag"],
                "removal_rate": row["removal_rate"],
                "avg_urls_per_request": row["avg_urls_per_request"],
                "major_org_requests": major_org_requests,
                "unique_major_orgs": unique_major_orgs,
                "major_org_ratio": major_org_ratio,
                "major_org_requests_last_90d": major_org_requests_last_90d,
                "major_studio_requests": major_studio_requests,
                "unique_major_studios": unique_major_studios,
                "has_takedown_history": True,
                "stats": stats,
            }
        )

    def get_takedown_stats(self, domain: str) -> Optional[TakedownStats]:
        """
        Get takedown statistics for a domain.

        Convenience method that returns TakedownStats object.

        Args:
            domain: Domain to look up

        Returns:
            TakedownStats object or None if no data
        """
        result = self.collect(domain)
        if result.success and result.data:
            return result.data.get("stats")
        return None

    def get_top_domains(self, n: int = 100) -> list[dict]:
        """
        Get top N most targeted domains.

        Args:
            n: Number of domains to return

        Returns:
            List of domain statistics dicts
        """
        if not self._loaded:
            self.load_data()

        if self.df is None:
            return []

        # Data is already sorted by total_requests descending
        top = self.df.head(n)

        return [
            {
                "domain": row["Domain"],
                "total_requests": row["total_requests"],
                "total_urls_removed": row["total_urls_removed"],
                "removal_rate": row["removal_rate"],
            }
            for row in top.iter_rows(named=True)
        ]

    def get_stats(self) -> dict:
        """Get statistics about loaded data."""
        if not self._loaded:
            self.load_data()

        if self.df is None:
            return {}

        return {
            "total_domains": len(self.df),
            "data_path": self.data_path,
            "memory_usage_mb": self.df.estimated_size() / (1024**2),
            "total_requests_all_domains": self.df["total_requests"].sum(),
            "total_urls_removed_all_domains": self.df["total_urls_removed"].sum(),
            "avg_removal_rate": self.df["removal_rate"].mean(),
        }
