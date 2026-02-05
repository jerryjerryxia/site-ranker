"""Tranco ranking list collector.

Tranco is a research-oriented ranking of top websites that combines
multiple data sources (Alexa, Umbrella, Majestic, etc.) for stability.

Website: https://tranco-list.eu/
"""

from pathlib import Path
from typing import Optional

from src.collectors.base import CollectorResult, LocalCollector
from src.utils.domain_utils import get_apex_domain, normalize_domain


class TrancoCollector(LocalCollector):
    """
    Collector for Tranco ranking data.

    Supports both apex domain and subdomain lookups.
    """

    def __init__(
        self,
        apex_list_path: Optional[Path] = None,
        subdomain_list_path: Optional[Path] = None,
    ):
        """
        Initialize Tranco collector.

        Args:
            apex_list_path: Path to tranco apex domain CSV
            subdomain_list_path: Path to tranco subdomain CSV (optional)
        """
        # Use default paths if not provided
        if apex_list_path is None:
            project_root = Path(__file__).parent.parent.parent
            apex_list_path = project_root / "tranco_08_12_2025.csv"

        if subdomain_list_path is None:
            project_root = Path(__file__).parent.parent.parent
            subdomain_list_path = project_root / "tranco_08_12_2025_sub.csv"

        super().__init__(name="Tranco", data_path=str(apex_list_path))

        self.apex_list_path = apex_list_path
        self.subdomain_list_path = subdomain_list_path

        # Caches: domain -> rank
        self.apex_cache: dict[str, int] = {}
        self.subdomain_cache: dict[str, int] = {}

    def load_data(self) -> None:
        """Load Tranco lists into memory."""
        # Load apex domain list
        if self.apex_list_path.exists():
            with open(self.apex_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Format: rank,domain
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        rank_str, domain = parts
                        try:
                            rank = int(rank_str)
                            domain = normalize_domain(domain)
                            self.apex_cache[domain] = rank
                        except ValueError:
                            continue

        # Load subdomain list if available
        if self.subdomain_list_path and self.subdomain_list_path.exists():
            with open(self.subdomain_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        rank_str, domain = parts
                        try:
                            rank = int(rank_str)
                            domain = normalize_domain(domain)
                            self.subdomain_cache[domain] = rank
                        except ValueError:
                            continue

        self._loaded = True

    def _lookup(self, normalized_domain: str) -> CollectorResult:
        """
        Look up domain rank in Tranco lists.

        Strategy:
        1. Try exact match in subdomain list (if available)
        2. Try exact match in apex list
        3. Try apex domain in apex list

        Args:
            normalized_domain: Normalized domain string

        Returns:
            CollectorResult with rank data
        """
        # Try subdomain list first (more specific)
        if normalized_domain in self.subdomain_cache:
            rank = self.subdomain_cache[normalized_domain]
            return CollectorResult(
                domain=normalized_domain,
                success=True,
                data={
                    "rank": rank,
                    "source": "subdomain_list",
                    "domain": normalized_domain,
                }
            )

        # Try apex list with exact match
        if normalized_domain in self.apex_cache:
            rank = self.apex_cache[normalized_domain]
            return CollectorResult(
                domain=normalized_domain,
                success=True,
                data={
                    "rank": rank,
                    "source": "apex_list",
                    "domain": normalized_domain,
                }
            )

        # Try extracting apex domain and looking up
        # BUT: Skip shared hosting platforms where subdomains are independent sites
        apex = get_apex_domain(normalized_domain)

        # Shared hosting platforms - subdomains are independent sites
        SHARED_HOSTING_PLATFORMS = {
            # Blogging platforms
            'blogspot.com', 'wordpress.com', 'tumblr.com', 'medium.com',
            'hatenablog.com', 'hateblo.jp', 'livejournal.com',

            # Website builders
            'wixsite.com', 'webflow.io', 'squarespace.com', 'weebly.com',
            'yolasite.com', 'framer.website', 'notion.site', 'carrd.co',

            # Free hosting
            'altervista.org', '000webhostapp.com', 'freehosting.com',
            '110mb.com', 'freewebhostingarea.com', 'uhostfull.com',

            # Cloud platforms
            'web.app', 'firebaseapp.com', 'herokuapp.com', 'azurewebsites.net',
            'netlify.app', 'vercel.app', 'pages.dev', 'onrender.com',
            'appspot.com', 'rhcloud.com', 'cloudfunctions.net',

            # Developer platforms
            'github.io', 'gitlab.io', 'bitbucket.io', 'repl.co', 'glitch.me',

            # CDN/Edge/Workers
            'cloudfront.net', 'surge.sh', 'now.sh', 'workers.dev',
            'akamaihd.net', 'fastly.net', 'edgecastcdn.net',
            'azurefd.net', 'azureedge.net',

            # AWS services (S3, CloudFront, etc. - all subdomains are independent)
            'amazonaws.com',  # Parent domain - covers all AWS services
            'storage.googleapis.com', 'blob.core.windows.net',

            # Dynamic DNS services (users get subdomains)
            'duckdns.org', 'ddns.net', 'freeddns.org', 'hopto.org',
            'dyndns.org', 'dyndns.tv', 'no-ip.org', 'freemyip.com',

            # Domain resellers (cheap second-level domains)
            'in.net', 'us.com', 'ru.com', 'it.com', 'uk.com', 'co.com',
            'eu.org', 'org.ru', 'net.ru', 'com.de', 'ru.net', 'pp.ua',
            'com.ru', 'com.se', 'eu.com', 'sa.com', 'za.com', 'cn.com',
            'pp.ru', 'webredirect.org',

            # E-commerce platforms
            'myshopify.com', 'bigcartel.com', 'tictail.com',

            # Google services
            'translate.goog', 'googleusercontent.com', 'googledrive.com',

            # Other platforms
            'zendesk.com', 'freshdesk.com', 'helpscoutdocs.com',
        }

        if apex != normalized_domain and apex in self.apex_cache and apex not in SHARED_HOSTING_PLATFORMS:
            rank = self.apex_cache[apex]
            return CollectorResult(
                domain=normalized_domain,
                success=True,
                data={
                    "rank": rank,
                    "source": "apex_list",
                    "domain": apex,
                    "queried_subdomain": normalized_domain,
                }
            )

        # Not found
        return CollectorResult(
            domain=normalized_domain,
            success=False,
            error="Domain not found in Tranco lists"
        )

    def get_rank(self, domain: str) -> Optional[int]:
        """
        Get Tranco rank for a domain.

        Convenience method that returns just the rank.

        Args:
            domain: Domain to look up

        Returns:
            Rank (1 = most popular) or None if not found
        """
        result = self.collect(domain)
        if result.success and result.data:
            return result.data.get("rank")
        return None

    def get_stats(self) -> dict:
        """Get statistics about loaded data."""
        if not self._loaded:
            self.load_data()

        return {
            "apex_domains_loaded": len(self.apex_cache),
            "subdomain_entries_loaded": len(self.subdomain_cache),
            "apex_list_path": str(self.apex_list_path),
            "subdomain_list_path": str(self.subdomain_list_path) if self.subdomain_list_path else None,
            "top_rank": min(self.apex_cache.values()) if self.apex_cache else None,
            "bottom_rank": max(self.apex_cache.values()) if self.apex_cache else None,
        }
