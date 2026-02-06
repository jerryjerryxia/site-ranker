#!/usr/bin/env python3
"""
Batch domain online status checker with content-aware classification.

Categories:
- Online: Site responds and appears functional
- Offline: DNS fails or site doesn't respond
- Parked: Domain parking page detected
- Blocked/Seized: Law enforcement seizure page
- Unknown: Couldn't determine

Usage:
    python scripts/check_online_status.py --token YOUR_IPINFO_TOKEN
    python scripts/check_online_status.py --token YOUR_IPINFO_TOKEN --limit 1000
"""

import argparse
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASETS

# Detection patterns
PARKED_INDICATORS = [
    # Domain parking services
    r'domain\s*(is\s*)?(for\s+)?sale',
    r'buy\s+this\s+domain',
    r'domain\s+parking',
    r'parked\s+(free|domain)',
    r'this\s+domain\s+(may\s+be|is)\s+(for\s+sale|available)',
    r'sedoparking',
    r'parkingcrew',
    r'bodis\.com',
    r'hugedomains',
    r'dan\.com',
    r'afternic',
    r'godaddy\s+auctions',
    r'namecheap.*marketplace',
    r'domainmarket',
    r'uniregistry',
    r'click\s+here\s+to\s+(buy|purchase|inquire)',
    r'domain\s+has\s+expired',
    r'renew\s+this\s+domain',
    r'sponsor(ed)?\s+listing',
    r'related\s+links',
    r'ParkLogic',
    r'Above\.com',
]

SEIZED_INDICATORS = [
    # Law enforcement
    r'seized\s+by',
    r'domain\s+(has\s+been\s+)?seized',
    r'this\s+domain\s+has\s+been\s+seized',
    r'fbi\.gov',
    r'justice\.gov',
    r'ice\.gov',
    r'europol',
    r'interpol',
    r'seized\s+pursuant',
    r'law\s+enforcement',
    r'criminal\s+investigation',
    r'department\s+of\s+justice',
    r'homeland\s+security',
    r'operation\s+\w+',  # Operation [Name]
    r'this\s+site\s+has\s+been\s+blocked',
    r'court\s+order',
    r'mpaa.*seized',
    r'riaa.*seized',
    r'takedown.*order',
    # Common seizure page text
    r'violat(ed?|ing)\s+(federal|copyright|intellectual\s+property)',
]

# Compile patterns for efficiency
PARKED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PARKED_INDICATORS]
SEIZED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SEIZED_INDICATORS]


def resolve_dns(domain: str, timeout: float = 5.0) -> str | None:
    """Resolve domain to IP address."""
    socket.setdefaulttimeout(timeout)
    try:
        return socket.gethostbyname(domain)
    except (socket.gaierror, socket.timeout):
        return None


def get_ipinfo(ip: str, token: str) -> dict:
    """Get IP info from IPinfo Lite API."""
    try:
        resp = requests.get(
            f"https://api.ipinfo.io/lite/{ip}",
            params={"token": token},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def fetch_page(domain: str, timeout: float = 10.0) -> tuple[int | None, str]:
    """Fetch page and return (status_code, content)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    for proto in ["https", "http"]:
        try:
            resp = requests.get(
                f"{proto}://{domain}",
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                verify=False  # Some piracy sites have bad certs
            )
            # Get text content, limit to first 50KB
            content = resp.text[:50000] if resp.text else ""
            return resp.status_code, content
        except requests.RequestException:
            continue
    
    return None, ""


def classify_content(content: str) -> str:
    """Classify page content as parked, seized, or online."""
    if not content:
        return "Unknown"
    
    content_lower = content.lower()
    
    # Check for seized indicators first (higher priority)
    seized_matches = sum(1 for p in SEIZED_PATTERNS if p.search(content_lower))
    if seized_matches >= 2:  # Require multiple matches for confidence
        return "Blocked/Seized"
    
    # Check for parked indicators
    parked_matches = sum(1 for p in PARKED_PATTERNS if p.search(content_lower))
    if parked_matches >= 2:  # Require multiple matches for confidence
        return "Parked"
    
    # If we got content and it's substantial, assume online
    # Filter out very short pages (likely errors or redirects)
    text_content = re.sub(r'<[^>]+>', '', content)  # Strip HTML
    text_content = re.sub(r'\s+', ' ', text_content).strip()
    
    if len(text_content) > 500:
        return "Online"
    elif len(text_content) > 100:
        # Short content, could be parking or minimal site
        if parked_matches >= 1:
            return "Parked"
        return "Online"
    else:
        return "Unknown"


def check_domain(domain: str, token: str) -> dict:
    """Check single domain's online status with content analysis."""
    result = {
        "domain": domain,
        "checked_at": datetime.now().isoformat(),
        "online_status": "Unknown",
        "ip": None,
        "http_status": None,
        "country": None,
        "country_code": None,
        "continent": None,
        "asn": None,
        "as_name": None,
        "as_domain": None,
        "error": None,
    }
    
    # Step 1: DNS resolution
    ip = resolve_dns(domain)
    if not ip:
        result["online_status"] = "Offline"
        result["error"] = "DNS_FAILED"
        return result
    
    result["ip"] = ip
    
    # Step 2: IPinfo lookup
    info = get_ipinfo(ip, token)
    if "error" not in info:
        result["country"] = info.get("country")
        result["country_code"] = info.get("country_code")
        result["continent"] = info.get("continent")
        result["asn"] = info.get("asn")
        result["as_name"] = info.get("as_name")
        result["as_domain"] = info.get("as_domain")
    
    # Step 3: HTTP fetch and content analysis
    status_code, content = fetch_page(domain)
    result["http_status"] = status_code
    
    if status_code is None:
        # DNS resolved but HTTP failed
        result["online_status"] = "Offline"
        result["error"] = "HTTP_FAILED"
        return result
    
    if status_code >= 400:
        # HTTP error
        if status_code == 403:
            # Could be blocked or Cloudflare protection
            result["online_status"] = "Unknown"
            result["error"] = f"HTTP_{status_code}"
        elif status_code == 404:
            result["online_status"] = "Offline"
            result["error"] = "HTTP_404"
        elif status_code >= 500:
            result["online_status"] = "Offline"
            result["error"] = f"HTTP_{status_code}"
        else:
            result["online_status"] = "Unknown"
            result["error"] = f"HTTP_{status_code}"
        return result
    
    # Step 4: Content classification
    result["online_status"] = classify_content(content)
    
    return result


def load_domains_from_gtr(limit: int | None = None) -> list[str]:
    """Load domains from the curated GTR dataset."""
    curated_path = DATASETS["video_piracy"]["path"]
    if not curated_path.exists():
        raise FileNotFoundError(f"GTR data not found: {curated_path}")
    
    df = pd.read_parquet(curated_path)
    domains = df["Domain"].dropna().unique().tolist()
    
    if limit:
        domains = domains[:limit]
    
    return domains


def load_domains_from_file(filepath: str) -> list[str]:
    """Load domains from a text file (one per line)."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Check domain online status via IPinfo + HTTP")
    parser.add_argument("--token", required=True, help="IPinfo API token")
    parser.add_argument("--domains", help="Path to domain list file (one per line)")
    parser.add_argument("--limit", type=int, help="Limit number of domains to check")
    parser.add_argument("--output", default="data/online_status.csv", help="Output CSV path")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--resume", action="store_true", help="Skip already-checked domains")
    args = parser.parse_args()
    
    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Load domains
    if args.domains:
        domains = load_domains_from_file(args.domains)
    else:
        domains = load_domains_from_gtr(args.limit)
    
    print(f"Loaded {len(domains)} domains to check")
    
    # Resume support - skip already checked
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checked = set()
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        checked = set(existing["domain"].tolist())
        print(f"Resuming: {len(checked)} already checked")
    
    domains = [d for d in domains if d not in checked]
    print(f"Checking {len(domains)} domains...")
    
    if not domains:
        print("Nothing to check!")
        return
    
    # Process with thread pool
    results = []
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_domain, d, args.token): d for d in domains}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            # Progress
            if i % 100 == 0 or i == len(domains):
                elapsed = time.time() - start
                rate = i / elapsed
                status_counts = {}
                for r in results:
                    s = r["online_status"]
                    status_counts[s] = status_counts.get(s, 0) + 1
                status_str = " | ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
                print(f"[{i}/{len(domains)}] {rate:.1f}/s | {status_str}")
    
    # Save results
    df = pd.DataFrame(results)
    
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\nDone! Results saved to {output_path}")
    print(f"\nStatus breakdown:")
    print(df["online_status"].value_counts().to_string())
    
    # Top hosting providers for online sites
    online_df = df[df["online_status"] == "Online"]
    if len(online_df) > 0 and "as_name" in df.columns:
        print("\nTop hosting providers (Online sites):")
        print(online_df["as_name"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
