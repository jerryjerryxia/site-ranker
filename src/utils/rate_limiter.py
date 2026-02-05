"""Rate limiting utilities for API and scraping requests."""

import asyncio
import time
from typing import Optional


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for controlling request rates.

    Allows bursts up to max_tokens, then enforces rate limit.
    Works with both sync and async code.

    Args:
        requests_per_second: Maximum sustained request rate
        burst_size: Maximum burst capacity (defaults to requests_per_second)

    Examples:
        >>> limiter = TokenBucketRateLimiter(requests_per_second=2)
        >>> limiter.acquire()  # First request - immediate
        >>> limiter.acquire()  # Second request - immediate
        >>> limiter.acquire()  # Third request - waits ~0.5s
    """

    def __init__(self, requests_per_second: float, burst_size: Optional[int] = None):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")

        self.rate = requests_per_second
        self.max_tokens = burst_size if burst_size is not None else max(1, int(requests_per_second))
        self.tokens = self.max_tokens
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time since last update."""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.rate

        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_update = now

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens synchronously, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        if tokens > self.max_tokens:
            raise ValueError(f"Cannot acquire {tokens} tokens (max: {self.max_tokens})")

        wait_time = 0.0

        while True:
            self._add_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return wait_time

            # Calculate wait time needed
            deficit = tokens - self.tokens
            sleep_time = deficit / self.rate

            time.sleep(sleep_time)
            wait_time += sleep_time

    async def acquire_async(self, tokens: int = 1) -> float:
        """
        Acquire tokens asynchronously, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        if tokens > self.max_tokens:
            raise ValueError(f"Cannot acquire {tokens} tokens (max: {self.max_tokens})")

        async with self._lock:
            wait_time = 0.0

            while True:
                self._add_tokens()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return wait_time

                # Calculate wait time needed
                deficit = tokens - self.tokens
                sleep_time = deficit / self.rate

                await asyncio.sleep(sleep_time)
                wait_time += sleep_time

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        self._add_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        self.tokens = self.max_tokens
        self.last_update = time.time()

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        self._add_tokens()
        return self.tokens


class FixedWindowRateLimiter:
    """
    Fixed window rate limiter - simpler alternative to token bucket.

    Limits requests to a fixed count per time window.

    Args:
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds

    Examples:
        >>> limiter = FixedWindowRateLimiter(max_requests=10, window_seconds=60)
        >>> limiter.acquire()  # Allowed
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: list[float] = []
        self._lock = asyncio.Lock()

    def _clean_old_requests(self) -> None:
        """Remove requests outside the current window."""
        now = time.time()
        cutoff = now - self.window_seconds
        self.request_times = [t for t in self.request_times if t > cutoff]

    def acquire(self) -> bool:
        """
        Try to acquire a request slot synchronously.

        Returns:
            True if request allowed, False if rate limit exceeded
        """
        self._clean_old_requests()

        if len(self.request_times) < self.max_requests:
            self.request_times.append(time.time())
            return True

        return False

    async def acquire_async(self) -> bool:
        """
        Try to acquire a request slot asynchronously.

        Returns:
            True if request allowed, False if rate limit exceeded
        """
        async with self._lock:
            self._clean_old_requests()

            if len(self.request_times) < self.max_requests:
                self.request_times.append(time.time())
                return True

            return False

    def wait_time(self) -> float:
        """
        Get time until next request slot available.

        Returns:
            Seconds to wait, or 0 if slot available now
        """
        self._clean_old_requests()

        if len(self.request_times) < self.max_requests:
            return 0.0

        # Wait until oldest request falls out of window
        oldest = min(self.request_times)
        return (oldest + self.window_seconds) - time.time()

    def reset(self) -> None:
        """Reset the limiter."""
        self.request_times = []
