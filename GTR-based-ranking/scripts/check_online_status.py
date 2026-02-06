#!/usr/bin/env python3
"""
Batch domain online status checker using IPinfo Lite API.

Usage:
    python scripts/check_online_status.py --token YOUR_IPINFO_TOKEN
    python scripts/check_online_status.py --token YOUR_IPINFO_TOKEN --limit 1000
    python scripts/check_online_status.py --token YOUR_IPINFO_TOKEN --domains domains.txt
"""

import argparse
import csv
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
from config import DATA_SOURCES


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


def check_domain(domain: str, token: str) -> dict:
    """Check single domain's online status and IP info."""
    result = {
        "domain": domain,
        "checked_at": datetime.now().isoformat(),
        "online": False,
        "ip": None,
        "country": None,
        "country_code": None,
        "continent": None,
        "asn": None,
        "as_name": None,
        "as_domain": None,
        "error": None,
    }
    
    # DNS resolution
    ip = resolve_dns(domain)
    if not ip:
        result["error"] = "DNS_FAILED"
        return result
    
    result["ip"] = ip
    result["online"] = True
    
    # IPinfo lookup
    info = get_ipinfo(ip, token)
    if "error" in info:
        result["error"] = info["error"]
    else:
        result["country"] = info.get("country")
        result["country_code"] = info.get("country_code")
        result["continent"] = info.get("continent")
        result["asn"] = info.get("asn")
        result["as_name"] = info.get("as_name")
        result["as_domain"] = info.get("as_domain")
    
    return result


def load_domains_from_gtr(limit: int | None = None) -> list[str]:
    """Load domains from the curated GTR dataset."""
    curated_path = Path(DATA_SOURCES["curated"]["path"]).expanduser()
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
    parser = argparse.ArgumentParser(description="Check domain online status via IPinfo")
    parser.add_argument("--token", required=True, help="IPinfo API token")
    parser.add_argument("--domains", help="Path to domain list file (one per line)")
    parser.add_argument("--limit", type=int, help="Limit number of domains to check")
    parser.add_argument("--output", default="data/online_status.csv", help="Output CSV path")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--resume", action="store_true", help="Skip already-checked domains")
    args = parser.parse_args()
    
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
                online = sum(1 for r in results if r["online"])
                print(f"[{i}/{len(domains)}] {rate:.1f}/s | Online: {online} ({100*online/i:.1f}%)")
    
    # Save results
    df = pd.DataFrame(results)
    
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    
    # Summary
    online = df["online"].sum()
    total = len(df)
    print(f"\nDone! Results saved to {output_path}")
    print(f"Online: {online}/{total} ({100*online/total:.1f}%)")
    
    # Top hosting providers
    if "as_name" in df.columns:
        print("\nTop hosting providers:")
        print(df[df["online"]]["as_name"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
