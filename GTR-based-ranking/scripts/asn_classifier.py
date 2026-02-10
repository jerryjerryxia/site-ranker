#!/usr/bin/env python3
"""
ASN-based domain classification.

Classifies domains by their hosting infrastructure:
- Parked: Domain parking/monetization services
- CDN-Protected: Behind major CDNs (Cloudflare, Akamai)
- Bulletproof: Known piracy-friendly/ignore-takedown hosters
- Major Cloud: AWS, Google, Hetzner, etc.
- Other Hosting: Unknown hosting, needs HTTP check

Usage:
    python scripts/asn_classifier.py --input data/dns_results.csv --output data/classified.csv
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

# ASN classification patterns (case-insensitive matching)
ASN_CATEGORIES = {
    "Parked": [
        "Trellian",
        "Team Internet",
        "SEDO",
        "Bodis",
        "ParkLogic", 
        "Above.com",
        "Oversee",
        "DomainSponsor",
        "Parking Crew",
        "Voodoo",
        "Fabulous",
        "HugeDomains",
        "Afternic",
        "Uniregistry",
        "Dan.com",
    ],
    "CDN-Protected": [
        "Cloudflare",
        "Akamai",
        "Fastly",
        "CloudFront",
        "StackPath",
        "KeyCDN",
        "Imperva",
        "Sucuri",
        "Incapsula",
        "BunnyCDN",
    ],
    "Bulletproof": [
        "ALEXHOST",
        "FlokiNET",
        "Private Layer",
        "Ecatel",
        "BlueAngelHost",
        "Shinjiru",
        "Novogara",
        "1984 ehf",
        "Bahnhof",
        "PRQ",
    ],
    "Major Cloud": [
        "Amazon.com",
        "Amazon Web Services",
        "Google LLC",
        "Google Cloud",
        "Microsoft Azure",
        "Microsoft Corporation",
        "DigitalOcean",
        "Hetzner",
        "OVH",
        "Linode",
        "Vultr",
        "Oracle Cloud",
        "IBM Cloud",
        "Alibaba",
        "Tencent",
    ],
    "Legitimate Hosting": [
        "GoDaddy",
        "Namecheap",
        "Hostinger",
        "Bluehost",
        "SiteGround",
        "DreamHost",
        "InMotion",
        "A2 Hosting",
        "HostGator",
        "WP Engine",
    ],
}

# Infer online status from ASN category
ASN_TO_ONLINE_STATUS = {
    "Parked": "Parked",
    "CDN-Protected": "Online",  # Behind CDN = actively maintained
    "Bulletproof": "Online",     # Bulletproof = definitely running
    "Major Cloud": "Online",     # Professional hosting = likely online
    "Legitimate Hosting": "Online",
    "Other Hosting": None,       # Needs HTTP check
    "Unknown": None,             # Needs HTTP check
}


def classify_asn(as_name: str) -> str:
    """Classify ASN into category."""
    if pd.isna(as_name) or not as_name:
        return "Unknown"
    
    as_lower = as_name.lower()
    
    for category, patterns in ASN_CATEGORIES.items():
        for pattern in patterns:
            if pattern.lower() in as_lower:
                return category
    
    return "Other Hosting"


def infer_online_status(asn_category: str, current_status: str = None) -> str:
    """Infer online status from ASN category."""
    inferred = ASN_TO_ONLINE_STATUS.get(asn_category)
    
    if inferred is not None:
        return inferred
    
    # Keep existing status if we can't infer
    return current_status if current_status else "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Classify domains by ASN")
    parser.add_argument("--input", required=True, help="Input CSV with DNS results")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--http-needed", help="Optional: output list of domains needing HTTP check")
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    # Handle both old (online boolean) and new format
    if 'online' in df.columns and 'online_status' not in df.columns:
        df['online_status'] = df['online'].map({True: 'DNS_OK', False: 'Offline'})
    
    print(f"Total domains: {len(df)}")
    
    # Classify ASNs
    print("Classifying by ASN...")
    df['asn_category'] = df['as_name'].apply(classify_asn)
    
    # Infer online status for DNS-OK domains
    dns_ok_mask = df['online_status'] != 'Offline'
    df.loc[dns_ok_mask, 'online_status'] = df.loc[dns_ok_mask].apply(
        lambda row: infer_online_status(row['asn_category'], row.get('online_status')),
        axis=1
    )
    
    # Keep Offline as Offline
    df.loc[df['online_status'] == 'Offline', 'online_status'] = 'Offline'
    
    # Update timestamp
    df['checked_at'] = datetime.now().isoformat()
    
    # Stats
    print("\n=== ASN Category Distribution ===")
    print(df['asn_category'].value_counts().to_string())
    
    print("\n=== Online Status Distribution ===")
    print(df['online_status'].value_counts().to_string())
    
    # Domains needing HTTP check
    needs_http = df[df['online_status'].isin(['Unknown', 'DNS_OK'])]
    print(f"\nDomains needing HTTP check: {len(needs_http)}")
    
    # Save results
    # Drop old 'online' column if present
    if 'online' in df.columns:
        df = df.drop(columns=['online'])
    
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    
    # Optionally save HTTP-needed list
    if args.http_needed:
        needs_http[['domain']].to_csv(args.http_needed, index=False, header=False)
        print(f"HTTP-needed domains saved to {args.http_needed}")


if __name__ == "__main__":
    main()
