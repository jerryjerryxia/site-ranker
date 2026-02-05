"""Base collector class for all data collectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.domain_utils import normalize_domain


@dataclass
class CollectorResult:
    """
    Standard result object returned by collectors.

    Attributes:
        domain: Normalized domain name
        success: Whether data collection succeeded
        data: Collected data (structure varies by collector)
        error: Error message if collection failed
    """

    domain: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.

    All collectors should inherit from this class and implement
    the collect() method.
    """

    def __init__(self, name: str):
        """
        Initialize the collector.

        Args:
            name: Human-readable name for this collector
        """
        self.name = name

    def _normalize_domain(self, domain: str) -> str:
        """
        Normalize domain before processing.

        Args:
            domain: Raw domain string

        Returns:
            Normalized domain
        """
        return normalize_domain(domain)

    @abstractmethod
    def collect(self, domain: str) -> CollectorResult:
        """
        Collect data for a given domain.

        This method should be implemented by each collector subclass.

        Args:
            domain: Domain to collect data for

        Returns:
            CollectorResult with success status and data

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.name} must implement collect()")

    async def collect_async(self, domain: str) -> CollectorResult:
        """
        Async version of collect() for collectors that support it.

        Default implementation falls back to synchronous collect().
        Override in subclasses that support async collection.

        Args:
            domain: Domain to collect data for

        Returns:
            CollectorResult with success status and data
        """
        return self.collect(domain)

    def collect_batch(self, domains: list[str]) -> dict[str, CollectorResult]:
        """
        Collect data for multiple domains.

        Default implementation calls collect() for each domain.
        Override in subclasses that support efficient batch collection.

        Args:
            domains: List of domains to collect data for

        Returns:
            Dict mapping domain -> CollectorResult
        """
        results = {}
        for domain in domains:
            normalized = self._normalize_domain(domain)
            results[normalized] = self.collect(domain)
        return results

    async def collect_batch_async(self, domains: list[str]) -> dict[str, CollectorResult]:
        """
        Async batch collection.

        Default implementation calls collect_async() for each domain.
        Override in subclasses that support efficient batch collection.

        Args:
            domains: List of domains to collect data for

        Returns:
            Dict mapping domain -> CollectorResult
        """
        results = {}
        for domain in domains:
            normalized = self._normalize_domain(domain)
            results[normalized] = await self.collect_async(domain)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class LocalCollector(BaseCollector):
    """
    Base class for collectors that work with local data files.

    These collectors don't require rate limiting or network access.
    Examples: Tranco, Umbrella, Majestic, preprocessed Google Transparency.
    """

    def __init__(self, name: str, data_path: str):
        """
        Initialize local collector.

        Args:
            name: Human-readable name
            data_path: Path to local data file
        """
        super().__init__(name)
        self.data_path = data_path
        self._loaded = False

    @abstractmethod
    def load_data(self) -> None:
        """
        Load data from local file into memory.

        Should be called once before collecting data.
        Implement caching strategy in subclasses.
        """
        raise NotImplementedError(f"{self.name} must implement load_data()")

    def collect(self, domain: str) -> CollectorResult:
        """
        Collect data for a domain from loaded local data.

        Ensures data is loaded before collection.

        Args:
            domain: Domain to look up

        Returns:
            CollectorResult
        """
        if not self._loaded:
            self.load_data()
            self._loaded = True

        normalized = self._normalize_domain(domain)
        return self._lookup(normalized)

    @abstractmethod
    def _lookup(self, normalized_domain: str) -> CollectorResult:
        """
        Look up normalized domain in loaded data.

        Args:
            normalized_domain: Normalized domain string

        Returns:
            CollectorResult
        """
        raise NotImplementedError(f"{self.name} must implement _lookup()")


class RemoteCollector(BaseCollector):
    """
    Base class for collectors that fetch data from remote sources.

    These collectors require rate limiting and error handling.
    Examples: TGStat, Reddit, crt.sh, WHOIS.
    """

    def __init__(self, name: str, rate_limit: Optional[float] = None):
        """
        Initialize remote collector.

        Args:
            name: Human-readable name
            rate_limit: Requests per second (None = no limit)
        """
        super().__init__(name)
        self.rate_limit = rate_limit
        self._rate_limiter = None

        if rate_limit:
            from src.utils.rate_limiter import TokenBucketRateLimiter
            self._rate_limiter = TokenBucketRateLimiter(rate_limit)

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting before making a request."""
        if self._rate_limiter:
            self._rate_limiter.acquire()

    async def _apply_rate_limit_async(self) -> None:
        """Apply rate limiting asynchronously."""
        if self._rate_limiter:
            await self._rate_limiter.acquire_async()
