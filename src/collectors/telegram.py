"""
Telegram channel statistics collector.

Uses multiple free strategies to collect Telegram channel subscriber counts:
1. Telegram Bot API (getChatMemberCount) - requires bot token
2. Web scraping from t.me public pages
3. TGStat public API (free tier)

For anti-piracy use case, many piracy sites advertise their Telegram channels,
so we combine domain scraping with channel statistics collection.
"""

import re
from typing import Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from src.collectors.base import CollectorResult, RemoteCollector
from src.config import config
from src.features.schema import TelegramChannel


class TelegramCollector(RemoteCollector):
    """
    Collector for Telegram channel statistics.

    Supports multiple strategies for fetching subscriber counts without paid APIs.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        rate_limit: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize Telegram collector.

        Args:
            bot_token: Optional Telegram Bot API token (defaults to config)
            rate_limit: Requests per second (defaults to config)
            timeout: HTTP request timeout in seconds (defaults to config)
        """
        # Use config defaults if not provided
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        rate_limit = rate_limit or config.TELEGRAM_RATE_LIMIT
        self.timeout = timeout or config.HTTP_TIMEOUT

        super().__init__(name="Telegram", rate_limit=rate_limit)
        self.client = httpx.Client(timeout=self.timeout, follow_redirects=True)

    def collect(self, channel_username: str) -> CollectorResult:
        """
        Collect statistics for a Telegram channel.

        Args:
            channel_username: Channel username (with or without @)

        Returns:
            CollectorResult with channel statistics
        """
        self._apply_rate_limit()

        # Normalize channel username
        channel = channel_username.lstrip('@')

        # Try strategies in order of reliability
        strategies = [
            self._try_bot_api,
            self._try_tgstat_scrape,
            self._try_web_preview,
        ]

        for strategy in strategies:
            try:
                result = strategy(channel)
                if result and result.success:
                    return result
            except Exception as e:
                # Continue to next strategy
                continue

        # All strategies failed
        return CollectorResult(
            domain=channel,
            success=False,
            error="Could not retrieve channel statistics from any source"
        )

    def _try_bot_api(self, channel: str) -> Optional[CollectorResult]:
        """
        Try getting subscriber count via Telegram Bot API.

        Requires bot token and bot must be member of channel.
        Implements exponential backoff for rate limiting.
        """
        if not self.bot_token:
            return None

        url = f"https://api.telegram.org/bot{self.bot_token}/getChatMemberCount"
        params = {"chat_id": f"@{channel}"}

        max_retries = 3
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
            try:
                response = self.client.get(url, params=params)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = response.json().get('parameters', {}).get('retry_after', retry_delay)
                    import time
                    time.sleep(retry_after)
                    retry_delay *= 2  # Exponential backoff
                    continue

                # Success
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        count = data.get("result", 0)
                        return CollectorResult(
                            domain=channel,
                            success=True,
                            data={
                                "channel_url": f"https://t.me/{channel}",
                                "subscribers": count,
                                "name": channel,
                                "method": "bot_api",
                            }
                        )

                # Other errors
                break

            except Exception:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                break

        return None

    def _try_tgstat_scrape(self, channel: str) -> Optional[CollectorResult]:
        """
        Try scraping TGStat public page.

        TGStat shows channel stats on public pages without auth.
        """
        url = f"https://tgstat.com/channel/@{channel}"

        try:
            response = self.client.get(url)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for subscriber count in meta tags or page content
            # TGStat typically shows subscriber count prominently

            # Try meta description
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                desc = meta_desc.get('content', '')
                # Look for numbers followed by "subscribers"
                match = re.search(r'([\d\s,]+)\s*subscribers?', desc, re.IGNORECASE)
                if match:
                    count_str = match.group(1).replace(',', '').replace(' ', '')
                    try:
                        count = int(count_str)
                        return CollectorResult(
                            domain=channel,
                            success=True,
                            data={
                                "channel_url": f"https://t.me/{channel}",
                                "subscribers": count,
                                "name": channel,
                                "method": "tgstat_scrape",
                            }
                        )
                    except ValueError:
                        pass

            # Try finding in page content
            # Look for elements with subscriber counts
            for elem in soup.find_all(['div', 'span', 'p']):
                text = elem.get_text()
                if 'subscriber' in text.lower():
                    # Extract number near "subscribers"
                    match = re.search(r'([\d\s,]+)', text)
                    if match:
                        count_str = match.group(1).replace(',', '').replace(' ', '')
                        try:
                            count = int(count_str)
                            if count > 0 and count < 100_000_000:  # Sanity check
                                return CollectorResult(
                                    domain=channel,
                                    success=True,
                                    data={
                                        "channel_url": f"https://t.me/{channel}",
                                        "subscribers": count,
                                        "name": channel,
                                        "method": "tgstat_scrape",
                                    }
                                )
                        except ValueError:
                            continue

        except Exception:
            pass

        return None

    def _try_web_preview(self, channel: str) -> Optional[CollectorResult]:
        """
        Try scraping t.me public preview page.

        Note: This often doesn't show subscriber count, but might show
        other metadata we can use as a proxy signal.
        """
        url = f"https://t.me/{channel}"

        try:
            response = self.client.get(url)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Check if channel exists (vs redirect to app)
            page_title = soup.find('title')
            if page_title and 'Telegram' in page_title.get_text():
                # Channel exists, but subscriber count may not be shown
                # Return success with 0 subscribers to indicate channel exists
                return CollectorResult(
                    domain=channel,
                    success=True,
                    data={
                        "channel_url": f"https://t.me/{channel}",
                        "subscribers": None,  # Unknown
                        "name": channel,
                        "method": "web_preview",
                        "exists": True,
                    }
                )

        except Exception:
            pass

        return None

    def collect_multiple(self, channel_usernames: list[str]) -> CollectorResult:
        """
        Collect statistics for multiple Telegram channels and aggregate them.

        This is useful when a domain has multiple Telegram channels.
        We aggregate to get total network size across all channels.

        Args:
            channel_usernames: List of channel usernames (with or without @)

        Returns:
            CollectorResult with aggregated statistics
        """
        if not channel_usernames:
            return CollectorResult(
                domain="multiple_channels",
                success=False,
                error="No channel usernames provided"
            )

        channels_data = []
        total_subscribers = 0
        successful_collections = 0
        failed_channels = []

        for username in channel_usernames:
            # Skip invite links (they can't be queried for stats without joining)
            if username.startswith("invite_"):
                failed_channels.append(username)
                continue

            result = self.collect(username)

            if result.success and result.data:
                # Extract subscriber count
                subs = result.data.get("subscribers")
                if subs is not None:
                    total_subscribers += subs
                    successful_collections += 1
                    channels_data.append({
                        "username": username,
                        "subscribers": subs,
                        "channel_url": result.data.get("channel_url"),
                        "method": result.data.get("method"),
                    })
                else:
                    # Channel exists but subscriber count unknown
                    channels_data.append({
                        "username": username,
                        "subscribers": None,
                        "channel_url": result.data.get("channel_url"),
                        "method": result.data.get("method"),
                        "exists": True,
                    })
            else:
                failed_channels.append(username)

        # Determine success
        success = successful_collections > 0

        return CollectorResult(
            domain="multiple_channels",
            success=success,
            data={
                "total_subscribers": total_subscribers,
                "successful_collections": successful_collections,
                "total_channels": len(channel_usernames),
                "channels": channels_data,
                "failed_channels": failed_channels,
                "primary_channel": channels_data[0] if channels_data else None,
            }
        )

    def search_channel_by_domain(self, domain: str) -> Optional[str]:
        """
        Search for Telegram channel associated with a domain.

        Strategies:
        1. Try common patterns: @domainname, @domain_official
        2. Search TGStat for domain mentions
        3. Scrape the domain's website for Telegram links

        Args:
            domain: Domain name to search

        Returns:
            Channel username or None
        """
        # Try common patterns
        domain_base = domain.split('.')[0]  # Get 'rapidgator' from 'rapidgator.net'

        patterns = [
            domain_base,
            f"{domain_base}_official",
            f"{domain_base}net" if '.net' in domain else domain_base,
            f"{domain_base}com" if '.com' in domain else domain_base,
        ]

        for pattern in patterns:
            result = self.collect(pattern)
            if result.success:
                return pattern

        return None

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()


def extract_telegram_links(html: str) -> list[str]:
    """
    Extract Telegram channel/group links from HTML content.

    Args:
        html: HTML content to parse

    Returns:
        List of Telegram channel usernames (without @)
    """
    # Match t.me links
    pattern = r'https?://t\.me/([a-zA-Z0-9_]+)'
    matches = re.findall(pattern, html)

    # Remove 's/' preview prefix if present
    channels = [m.replace('s/', '') for m in matches]

    # Remove duplicates
    return list(set(channels))
