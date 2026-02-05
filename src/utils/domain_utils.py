"""Domain utilities for normalization, parsing, and validation."""

import re
from typing import Optional
from urllib.parse import urlparse

import tldextract

# MENA region TLDs
MENA_TLDS = {
    "eg",  # Egypt
    "sa",  # Saudi Arabia
    "ae",  # United Arab Emirates
    "ma",  # Morocco
    "tn",  # Tunisia
    "dz",  # Algeria
    "ly",  # Libya
    "jo",  # Jordan
    "lb",  # Lebanon
    "iq",  # Iraq
    "kw",  # Kuwait
    "qa",  # Qatar
    "bh",  # Bahrain
    "om",  # Oman
    "ye",  # Yemen
    "ps",  # Palestine
    "sy",  # Syria
}

# Arabic Unicode ranges
ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def normalize_domain(domain: str) -> str:
    """
    Normalize a domain to a canonical form.

    - Converts to lowercase
    - Strips www. prefix
    - Handles URLs by extracting domain
    - Handles IDN (internationalized domain names)
    - Strips whitespace and trailing dots

    Args:
        domain: Raw domain string or URL

    Returns:
        Normalized domain string

    Examples:
        >>> normalize_domain("WWW.EXAMPLE.COM")
        'example.com'
        >>> normalize_domain("https://www.example.com/path")
        'example.com'
        >>> normalize_domain("  example.com.  ")
        'example.com'
    """
    if not domain:
        return ""

    # Strip whitespace
    domain = domain.strip()

    # If it looks like a URL, parse it
    if "://" in domain:
        parsed = urlparse(domain)
        domain = parsed.netloc or parsed.path

    # Convert to lowercase
    domain = domain.lower()

    # Strip trailing dots
    domain = domain.rstrip(".")

    # Remove www. prefix
    if domain.startswith("www."):
        domain = domain[4:]

    # Handle IDN (internationalized domains)
    try:
        # Try to decode if it's already ASCII-encoded
        domain = domain.encode("ascii").decode("idna")
    except (UnicodeError, UnicodeDecodeError):
        # If that fails, try encoding to ASCII
        try:
            domain = domain.encode("idna").decode("ascii")
        except (UnicodeError, UnicodeDecodeError):
            # If all else fails, leave as is
            pass

    return domain


def extract_tld(domain: str) -> Optional[str]:
    """
    Extract the top-level domain (TLD) from a domain.

    Args:
        domain: Domain name

    Returns:
        TLD string (without dot) or None if invalid

    Examples:
        >>> extract_tld("example.com")
        'com'
        >>> extract_tld("example.co.uk")
        'uk'
        >>> extract_tld("subdomain.example.sa")
        'sa'
    """
    normalized = normalize_domain(domain)
    if not normalized:
        return None

    result = tldextract.extract(normalized)
    return result.suffix.split(".")[-1] if result.suffix else None


def is_mena_tld(domain: str) -> bool:
    """
    Check if domain uses a MENA (Middle East & North Africa) TLD.

    Args:
        domain: Domain name

    Returns:
        True if domain has a MENA TLD

    Examples:
        >>> is_mena_tld("example.sa")
        True
        >>> is_mena_tld("example.ae")
        True
        >>> is_mena_tld("example.com")
        False
    """
    tld = extract_tld(domain)
    return tld in MENA_TLDS if tld else False


def extract_subdomain(domain: str) -> Optional[str]:
    """
    Extract the subdomain portion of a domain.

    Args:
        domain: Domain name

    Returns:
        Subdomain string or None if no subdomain

    Examples:
        >>> extract_subdomain("www.example.com")
        'www'
        >>> extract_subdomain("api.staging.example.com")
        'api.staging'
        >>> extract_subdomain("example.com")
        None
    """
    normalized = normalize_domain(domain)
    if not normalized:
        return None

    result = tldextract.extract(normalized)
    return result.subdomain if result.subdomain else None


def get_apex_domain(domain: str) -> str:
    """
    Get the apex/registered domain (domain + TLD, no subdomains).

    Args:
        domain: Domain name

    Returns:
        Apex domain

    Examples:
        >>> get_apex_domain("www.example.com")
        'example.com'
        >>> get_apex_domain("api.staging.example.co.uk")
        'example.co.uk'
    """
    normalized = normalize_domain(domain)
    if not normalized:
        return ""

    result = tldextract.extract(normalized)
    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}"
    return normalized


def contains_arabic(text: str) -> bool:
    """
    Check if text contains Arabic characters.

    Args:
        text: Text to check

    Returns:
        True if text contains Arabic script

    Examples:
        >>> contains_arabic("مثال")
        True
        >>> contains_arabic("example")
        False
        >>> contains_arabic("example مثال mixed")
        True
    """
    return bool(ARABIC_PATTERN.search(text))


def is_valid_domain(domain: str) -> bool:
    """
    Check if a string is a valid domain name.

    Args:
        domain: String to validate

    Returns:
        True if valid domain format

    Examples:
        >>> is_valid_domain("example.com")
        True
        >>> is_valid_domain("sub.example.com")
        True
        >>> is_valid_domain("not a domain")
        False
        >>> is_valid_domain("")
        False
    """
    if not domain:
        return False

    normalized = normalize_domain(domain)

    # Basic pattern check
    pattern = re.compile(
        r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$",
        re.IGNORECASE
    )

    return bool(pattern.match(normalized))


def batch_normalize_domains(domains: list[str]) -> list[str]:
    """
    Normalize a batch of domains efficiently.

    Args:
        domains: List of domain strings

    Returns:
        List of normalized domains (duplicates removed, empty strings filtered)
    """
    normalized = {normalize_domain(d) for d in domains if d}
    return sorted([d for d in normalized if d])
