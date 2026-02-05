"""
Domain Infrastructure Checker
Batch checks DNS resolution, HTTP status, and CDN detection for domains.
Run as: python domain_checker.py [--limit 1000] [--output results.parquet]
"""

import asyncio
import socket
import ssl
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse

import httpx
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class DomainStatus:
    domain: str
    dns_resolves: bool
    ip_address: Optional[str]
    http_status: Optional[int]
    https_status: Optional[int]
    final_url: Optional[str]
    is_redirect: bool
    cdn_provider: Optional[str]
    is_parked: bool
    is_blocked: bool
    error: Optional[str]
    checked_at: str
    check_duration_ms: int


async def check_dns(domain: str) -> tuple[bool, Optional[str]]:
    """Check if domain resolves via DNS."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.getaddrinfo(domain, None, family=socket.AF_INET)
        if result:
            ip = result[0][4][0]
            return True, ip
        return False, None
    except (socket.gaierror, OSError):
        return False, None


async def check_http(client: httpx.AsyncClient, domain: str) -> dict:
    """Check HTTP/HTTPS status and detect CDN from headers."""
    result = {
        'http_status': None,
        'https_status': None,
        'final_url': None,
        'is_redirect': False,
        'cdn_provider': None,
        'is_parked': False,
        'is_blocked': False,
        'error': None,
    }
    
    for scheme in ['https', 'http']:
        url = f"{scheme}://{domain}"
        try:
            response = await client.head(url, follow_redirects=True, timeout=10.0)
            status = response.status_code
            
            if scheme == 'https':
                result['https_status'] = status
            else:
                result['http_status'] = status
            
            result['final_url'] = str(response.url)
            result['is_redirect'] = str(response.url).lower() != url.lower()
            
            # Detect CDN from headers
            server = response.headers.get('server', '').lower()
            cf_ray = response.headers.get('cf-ray')
            
            if cf_ray or 'cloudflare' in server:
                result['cdn_provider'] = 'cloudflare'
            elif 'akamai' in server:
                result['cdn_provider'] = 'akamai'
            elif 'fastly' in server:
                result['cdn_provider'] = 'fastly'
            elif 'cloudfront' in server:
                result['cdn_provider'] = 'cloudfront'
            elif 'ddos-guard' in server:
                result['cdn_provider'] = 'ddos-guard'
            
            if status in [403, 451]:
                result['is_blocked'] = True
            
            if scheme == 'https' and status:
                break
                
        except httpx.TimeoutException:
            result['error'] = 'timeout'
        except httpx.ConnectError:
            result['error'] = 'connect_error'
        except ssl.SSLError:
            result['error'] = 'ssl_error'
        except Exception as e:
            result['error'] = str(type(e).__name__)
    
    return result


async def check_domain(client: httpx.AsyncClient, domain: str) -> DomainStatus:
    """Full check for a single domain."""
    start = datetime.now()
    
    dns_resolves, ip_address = await check_dns(domain)
    
    http_result = {
        'http_status': None, 'https_status': None, 'final_url': None,
        'is_redirect': False, 'cdn_provider': None, 'is_parked': False,
        'is_blocked': False, 'error': None,
    }
    
    if dns_resolves:
        http_result = await check_http(client, domain)
    else:
        http_result['error'] = 'no_dns'
    
    duration = int((datetime.now() - start).total_seconds() * 1000)
    
    return DomainStatus(
        domain=domain,
        dns_resolves=dns_resolves,
        ip_address=ip_address,
        http_status=http_result['http_status'],
        https_status=http_result['https_status'],
        final_url=http_result['final_url'],
        is_redirect=http_result['is_redirect'],
        cdn_provider=http_result['cdn_provider'],
        is_parked=http_result['is_parked'],
        is_blocked=http_result['is_blocked'],
        error=http_result['error'],
        checked_at=datetime.now().isoformat(),
        check_duration_ms=duration,
    )


async def check_domains_batch(domains: list[str], concurrency: int = 100) -> list[DomainStatus]:
    """Check multiple domains with controlled concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=15.0,
        limits=httpx.Limits(max_connections=concurrency),
        headers={'User-Agent': 'Mozilla/5.0 (compatible; DomainChecker/1.0)'}
    ) as client:
        
        async def check_with_semaphore(domain: str) -> DomainStatus:
            async with semaphore:
                return await check_domain(client, domain)
        
        tasks = [check_with_semaphore(d) for d in domains]
        total = len(tasks)
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    return results


def derive_online_status(row: pd.Series) -> str:
    """Derive human-readable online status from check results."""
    if not row['dns_resolves']:
        return 'Offline (No DNS)'
    
    status = row['https_status'] or row['http_status']
    
    if pd.isna(status):
        return 'No Response'
    
    status = int(status)
    if status == 200:
        if row['is_parked']:
            return 'Parked'
        return 'Online'
    elif status in [301, 302, 303, 307, 308]:
        return 'Redirect'
    elif status == 403:
        return 'Forbidden'
    elif status == 404:
        return 'Not Found'
    elif status == 451:
        return 'Blocked (Legal)'
    elif status >= 500:
        return 'Server Error'
    else:
        return f'HTTP {status}'


async def main():
    parser = argparse.ArgumentParser(description='Check domain infrastructure status')
    parser.add_argument('--input', type=str, help='Input parquet file with domains')
    parser.add_argument('--output', type=str, default='domain_status.parquet', help='Output parquet file')
    parser.add_argument('--limit', type=int, help='Limit number of domains to check')
    parser.add_argument('--concurrency', type=int, default=100, help='Concurrent requests')
    parser.add_argument('--priority', action='store_true', help='Only check priority domains (active + high volume)')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent / "data" / "processed"
    input_path = args.input or (base_path / "video_piracy_clean.parquet")
    
    print(f"Loading domains from {input_path}...")
    df = pd.read_parquet(input_path)
    
    if args.priority:
        df['last_major_org_date'] = pd.to_datetime(df['last_major_org_date'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
        priority_mask = (df['last_major_org_date'] >= cutoff) | (df['total_urls_removed'] >= 100_000)
        df = df[priority_mask]
        print(f"Filtered to {len(df)} priority domains")
    
    domains = df['Domain'].tolist()
    
    if args.limit:
        domains = domains[:args.limit]
    
    print(f"Checking {len(domains)} domains with concurrency={args.concurrency}...")
    start_time = datetime.now()
    
    results = await check_domains_batch(domains, concurrency=args.concurrency)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Completed in {elapsed:.1f}s ({len(domains)/elapsed:.1f} domains/sec)")
    
    results_df = pd.DataFrame([asdict(r) for r in results])
    results_df['online_status'] = results_df.apply(derive_online_status, axis=1)
    
    print("\n=== Results Summary ===")
    print(results_df['online_status'].value_counts())
    print(f"\nCDN Distribution:")
    print(results_df['cdn_provider'].value_counts(dropna=False))
    
    output_path = Path(args.output)
    results_df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return results_df


if __name__ == '__main__':
    asyncio.run(main())
