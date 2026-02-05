"""
Site scraper to extract Telegram links and metadata from domains.

For piracy sites, Telegram channels are often advertised prominently
for content distribution, updates, and community engagement.
"""

import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.collectors.base import CollectorResult, RemoteCollector
from src.features.schema import SiteMetadata
from src.utils.domain_utils import contains_arabic, normalize_domain


class SiteScraperCollector(RemoteCollector):
    """
    Collector that scrapes domains to extract Telegram links and metadata.
    """

    def __init__(
        self,
        rate_limit: float = 0.5,  # 0.5 requests per second (respectful)
        timeout: int = 15,
        max_pages: int = 8,  # Check homepage + 7 other pages (expanded for better coverage)
    ):
        """
        Initialize site scraper.

        Args:
            rate_limit: Requests per second
            timeout: HTTP request timeout
            max_pages: Maximum pages to scrape per domain
        """
        super().__init__(name="SiteScraper", rate_limit=rate_limit)
        self.timeout = timeout
        self.max_pages = max_pages

        # User agent to avoid bot blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def collect(self, domain: str) -> CollectorResult:
        """
        Scrape domain for Telegram links and metadata.

        Args:
            domain: Domain to scrape

        Returns:
            CollectorResult with site metadata
        """
        self._apply_rate_limit()

        normalized = normalize_domain(domain)

        # Pages to check (in order of likelihood)
        # Expanded from 4 to 10 pages for better coverage
        urls_to_check = [
            f"https://{normalized}",              # Homepage
            f"https://{normalized}/about",        # About page
            f"https://{normalized}/contact",      # Contact page
            f"https://{normalized}/links",        # Links/Social page
            f"https://{normalized}/community",    # Community page
            f"https://{normalized}/telegram",     # Direct Telegram page
            f"https://{normalized}/social",       # Social media page
            f"https://{normalized}/join",         # Join/Subscribe page
            f"https://{normalized}/vip",          # VIP/Premium page
            f"https://{normalized}/premium",      # Premium page
        ]

        telegram_links = set()
        is_arabic = False
        detected_language = None
        uses_cloudflare = False

        for url in urls_to_check[:self.max_pages]:
            try:
                self._apply_rate_limit()  # Rate limit each request

                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    headers=self.headers
                ) as client:
                    response = client.get(url)

                    # Check for Cloudflare
                    if 'cf-ray' in response.headers or 'cloudflare' in response.text.lower()[:1000]:
                        uses_cloudflare = True

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Extract Telegram links
                        links = self._extract_telegram_links(response.text)
                        telegram_links.update(links)

                        # Detect language
                        if not is_arabic:
                            # Check HTML lang attribute
                            html_tag = soup.find('html')
                            if html_tag and html_tag.get('lang'):
                                lang = html_tag.get('lang', '').lower()
                                if 'ar' in lang:
                                    detected_language = 'ar'
                                    is_arabic = True
                                else:
                                    detected_language = lang[:2]

                            # Check for Arabic characters in text
                            page_text = soup.get_text()[:5000]  # First 5000 chars
                            if contains_arabic(page_text):
                                is_arabic = True
                                detected_language = 'ar'

            except httpx.HTTPStatusError:
                continue  # Page not found, try next
            except httpx.RequestError:
                continue  # Connection error, try next
            except Exception:
                continue  # Any other error, try next

        # Prepare metadata
        metadata = SiteMetadata(
            telegram_links=list(telegram_links),
            detected_language=detected_language,
            is_arabic=is_arabic,
            uses_cloudflare=uses_cloudflare,
        )

        return CollectorResult(
            domain=normalized,
            success=True,
            data={
                "telegram_links": list(telegram_links),
                "has_telegram": len(telegram_links) > 0,
                "telegram_count": len(telegram_links),
                "is_arabic": is_arabic,
                "detected_language": detected_language,
                "uses_cloudflare": uses_cloudflare,
                "metadata": metadata,
            }
        )

    def _extract_telegram_links(self, html: str) -> set[str]:
        """
        Extract Telegram channel/group links from HTML.

        Enhanced to detect:
        - Public channel links (t.me/channel)
        - Invite links (t.me/joinchat/...)
        - Alternative domains (telegram.me)
        - Protocol links (tg://)
        - @username mentions near "telegram"

        Args:
            html: HTML content

        Returns:
            Set of Telegram channel usernames and invite codes
        """
        channels = set()

        # Pattern 1: Standard t.me links
        pattern1 = r'https?://t\.me/([a-zA-Z0-9_]+)'
        matches1 = re.findall(pattern1, html, re.IGNORECASE)
        channels.update(m for m in matches1 if m not in ('s', 'joinchat'))  # Exclude preview/invite prefix

        # Pattern 2: Invite links (t.me/joinchat/XXXXX or t.me/+XXXXX)
        pattern2_joinchat = r'https?://t\.me/joinchat/([A-Za-z0-9_-]+)'
        matches2 = re.findall(pattern2_joinchat, html, re.IGNORECASE)
        channels.update(f"invite_{m}" for m in matches2)  # Prefix to distinguish from usernames

        pattern2_plus = r'https?://t\.me/\+([A-Za-z0-9_-]+)'
        matches2_plus = re.findall(pattern2_plus, html, re.IGNORECASE)
        channels.update(f"invite_{m}" for m in matches2_plus)

        # Pattern 3: Alternative domain (telegram.me)
        pattern3 = r'https?://telegram\.me/([a-zA-Z0-9_]+)'
        matches3 = re.findall(pattern3, html, re.IGNORECASE)
        channels.update(matches3)

        # Pattern 4: Telegram protocol links
        pattern4 = r'tg://resolve\?domain=([a-zA-Z0-9_]+)'
        matches4 = re.findall(pattern4, html, re.IGNORECASE)
        channels.update(matches4)

        # Pattern 5: @username mentions (only if near "telegram")
        pattern5 = r'@([a-zA-Z0-9_]{5,32})'  # Telegram usernames: 5-32 chars
        matches5 = re.findall(pattern5, html)
        for match in matches5:
            context_start = max(0, html.find(f'@{match}') - 150)
            context_end = min(len(html), html.find(f'@{match}') + 150)
            context = html[context_start:context_end].lower()
            if 'telegram' in context or 't.me' in context or 'tg://' in context:
                channels.add(match)

        return channels

    def get_telegram_channel(self, domain: str) -> Optional[str]:
        """
        Get the primary Telegram channel for a domain.

        Convenience method that returns the first found channel.

        Args:
            domain: Domain to check

        Returns:
            Telegram channel username or None
        """
        result = self.collect(domain)
        if result.success and result.data and result.data.get("telegram_links"):
            return result.data["telegram_links"][0]
        return None
