#!/usr/bin/env python3
"""
HTTP + content check for domains that already passed DNS.
Uses existing DNS results and adds content classification.
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Detection patterns
PARKED_INDICATORS = [
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
    r'operation\s+\w+',
    r'this\s+site\s+has\s+been\s+blocked',
    r'court\s+order',
    r'mpaa.*seized',
    r'riaa.*seized',
    r'takedown.*order',
    r'violat(ed?|ing)\s+(federal|copyright|intellectual\s+property)',
]

PARKED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PARKED_INDICATORS]
SEIZED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SEIZED_INDICATORS]


def fetch_page(domain: str, timeout: float = 10.0) -> tuple[int | None, str]:
    """Fetch page and return (status_code, content)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    for proto in ["https", "http"]:
        try:
            resp = requests.get(
                f"{proto}://{domain}",
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                verify=False
            )
            return resp.status_code, resp.text[:50000] if resp.text else ""
        except:
            continue
    return None, ""


def classify_content(content: str) -> str:
    """Classify page content."""
    if not content:
        return "Unknown"
    
    content_lower = content.lower()
    
    if sum(1 for p in SEIZED_PATTERNS if p.search(content_lower)) >= 2:
        return "Blocked/Seized"
    
    if sum(1 for p in PARKED_PATTERNS if p.search(content_lower)) >= 2:
        return "Parked"
    
    text = re.sub(r'<[^>]+>', '', content)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) > 500:
        return "Online"
    elif len(text) > 100:
        if sum(1 for p in PARKED_PATTERNS if p.search(content_lower)) >= 1:
            return "Parked"
        return "Online"
    return "Unknown"


def check_domain(domain: str) -> dict:
    """HTTP check single domain."""
    status_code, content = fetch_page(domain)
    
    if status_code is None:
        return {"domain": domain, "online_status": "Offline", "http_status": None, "error": "HTTP_FAILED"}
    
    if status_code >= 400:
        if status_code in [403]:
            return {"domain": domain, "online_status": "Unknown", "http_status": status_code, "error": f"HTTP_{status_code}"}
        elif status_code in [404] or status_code >= 500:
            return {"domain": domain, "online_status": "Offline", "http_status": status_code, "error": f"HTTP_{status_code}"}
        return {"domain": domain, "online_status": "Unknown", "http_status": status_code, "error": f"HTTP_{status_code}"}
    
    return {"domain": domain, "online_status": classify_content(content), "http_status": status_code, "error": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original DNS results CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    
    import urllib3
    urllib3.disable_warnings()
    
    # Load original DNS results
    df = pd.read_csv(args.input)
    dns_ok = df[df['online'] == True].copy()
    dns_fail = df[df['online'] == False].copy()
    
    print(f"DNS OK: {len(dns_ok)} | DNS Fail: {len(dns_fail)}")
    print(f"\nRunning HTTP checks on {len(dns_ok)} domains...")
    
    # HTTP check DNS-OK domains
    results = []
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_domain, d): d for d in dns_ok['domain'].tolist()}
        
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            
            if i % 200 == 0 or i == len(dns_ok):
                elapsed = time.time() - start
                rate = i / elapsed
                counts = {}
                for r in results:
                    s = r["online_status"]
                    counts[s] = counts.get(s, 0) + 1
                print(f"[{i}/{len(dns_ok)}] {rate:.1f}/s | " + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    
    # Merge HTTP results back
    http_df = pd.DataFrame(results)
    dns_ok = dns_ok.drop(columns=['online'], errors='ignore')
    dns_ok = dns_ok.merge(http_df, on='domain', how='left')
    
    # Update DNS fail with proper status
    dns_fail['online_status'] = 'Offline'
    dns_fail['http_status'] = None
    dns_fail['error'] = 'DNS_FAILED'
    dns_fail = dns_fail.drop(columns=['online'], errors='ignore')
    
    # Combine and save
    final = pd.concat([dns_ok, dns_fail], ignore_index=True)
    final['checked_at'] = datetime.now().isoformat()
    final.to_csv(args.output, index=False)
    
    print(f"\nDone! Saved to {args.output}")
    print(f"\nStatus breakdown:")
    print(final['online_status'].value_counts().to_string())


if __name__ == "__main__":
    main()
