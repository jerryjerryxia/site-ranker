#!/usr/bin/env python3
"""
HTTP check for DNS_OK domains only.
Updates the main online_status.csv in place.
Saves progress every N domains to survive interruptions.
"""

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import urllib3
urllib3.disable_warnings()

PARKED_PATTERNS = [re.compile(p, re.I) for p in [
    r'domain\s*(is\s*)?(for\s+)?sale', r'buy\s+this\s+domain', r'domain\s+parking',
    r'parked', r'sedoparking', r'parkingcrew', r'bodis', r'hugedomains', r'dan\.com',
    r'godaddy\s+auctions', r'domain\s+has\s+expired', r'related\s+links',
]]

SEIZED_PATTERNS = [re.compile(p, re.I) for p in [
    r'seized\s+by', r'domain\s+(has\s+been\s+)?seized', r'fbi\.gov', r'justice\.gov',
    r'europol', r'interpol', r'law\s+enforcement', r'department\s+of\s+justice',
]]


def fetch_and_classify(domain: str) -> dict:
    """Fetch page and classify content."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
    
    for proto in ["https", "http"]:
        try:
            r = requests.get(f"{proto}://{domain}", timeout=12, headers=headers, 
                           allow_redirects=True, verify=False)
            content = r.text[:50000].lower()
            
            if r.status_code >= 400:
                if r.status_code in [403, 405]:
                    return {"online_status": "Unknown", "http_status": r.status_code}
                return {"online_status": "Offline", "http_status": r.status_code}
            
            # Classify content
            if sum(1 for p in SEIZED_PATTERNS if p.search(content)) >= 2:
                return {"online_status": "Blocked/Seized", "http_status": r.status_code}
            if sum(1 for p in PARKED_PATTERNS if p.search(content)) >= 2:
                return {"online_status": "Parked", "http_status": r.status_code}
            
            text = re.sub(r'<[^>]+>', '', content)
            if len(text) > 300:
                return {"online_status": "Online", "http_status": r.status_code}
            return {"online_status": "Unknown", "http_status": r.status_code}
            
        except Exception:
            continue
    
    return {"online_status": "Offline", "http_status": None}


def main():
    csv_path = Path(__file__).parent.parent / "data" / "online_status.csv"
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get DNS_OK domains
    dns_ok_mask = df['online_status'] == 'DNS_OK'
    domains_to_check = df.loc[dns_ok_mask, 'domain'].tolist()
    
    print(f"DNS_OK domains to check: {len(domains_to_check)}")
    
    if not domains_to_check:
        print("Nothing to check!")
        return
    
    # Process with saves every 200
    SAVE_EVERY = 200
    WORKERS = 10
    
    results = {}
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch_and_classify, d): d for d in domains_to_check}
        
        for i, future in enumerate(as_completed(futures), 1):
            domain = futures[future]
            result = future.result()
            results[domain] = result
            
            if i % SAVE_EVERY == 0 or i == len(domains_to_check):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                
                # Count statuses
                counts = {}
                for r in results.values():
                    s = r['online_status']
                    counts[s] = counts.get(s, 0) + 1
                
                print(f"[{i}/{len(domains_to_check)}] {rate:.1f}/s | " + 
                      " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
                
                # Update dataframe and save
                for d, r in results.items():
                    mask = df['domain'] == d
                    df.loc[mask, 'online_status'] = r['online_status']
                    df.loc[mask, 'http_status'] = r.get('http_status')
                
                df['checked_at'] = datetime.now().isoformat()
                df.to_csv(csv_path, index=False)
                print(f"  [saved checkpoint]")
    
    print(f"\nDone! Final status distribution:")
    print(df['online_status'].value_counts().to_string())


if __name__ == "__main__":
    main()
