# DACIS - Domain Activity & Coverage Intelligence System

A Streamlit dashboard for analyzing piracy domains using Google Transparency Report (GTR) data and tracking Vobile's coverage across VideoTracker (VT) and VobileOne (V1) platforms.

## 🌐 Live Demo

**Production:** [gtr-site-ranker.streamlit.app](https://gtr-site-ranker.streamlit.app/)

## Overview

DACIS transforms Google Transparency Report data into actionable intelligence for piracy domain analysis. It provides:

- **Domain Intelligence**: Analyze 50K+ video piracy domains with lifetime notice counts, trends, and activity status
- **Coverage Tracking**: See which domains are covered by VideoTracker (VT) and VobileOne (V1)
- **Online Status**: HTTP-based checks to verify if domains are still active
- **Ad-hoc Analysis**: Upload CSV files to quickly check domain lists against our data

## Features

### 📋 Domains Tab
- Browse the full domain database with filtering and sorting
- Filter by: Status (Active/Inactive/Declining), Trend, Volume, Online status
- Sort by: URLs Removed, Last 30d activity, Studio count
- Export filtered results to CSV
- Interactive charts showing distribution and trends

### 📤 Upload Tab
- Upload a CSV or TXT file with a domain list
- Automatic domain cleaning (extracts root domain from URLs)
- Instant enrichment with GTR data and coverage status
- Download enriched results

### 🔎 Lookup Tab
- Deep-dive into individual domains
- View full notice history and trends
- Check coverage and online status

### 🗺️ Coverage Tab
- View all domains in Vobile's coverage map
- Filter by source: VT (VideoTracker), V1 (VobileOne), or All
- See series and episode counts per domain
- Track active vs inactive coverage
- Export coverage data

## Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/VobileLA/GTR-based-ranking.git
cd GTR-based-ranking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The dashboard opens at http://localhost:8501

### Using the Run Script

```bash
./run.sh
```

## Data Sources

| File | Description |
|------|-------------|
| `sample_data.parquet` | GTR domain data (~50K video piracy domains) |
| `data/Coverage_Sites.xlsx` | Combined VT + V1 coverage data |
| `data/VT_LSR_Site.xlsx` | VideoTracker coverage |
| `data/V1_Sites.xlsx` | VobileOne coverage |
| `data/online_status.csv` | HTTP status check results |

### Dataset Toggle

The app defaults to the curated video piracy subset (~50K domains) for fast loading. Toggle "Full dataset (6M)" in the sidebar to load the complete GTR dataset when needed.

## Scripts

Utility scripts for data maintenance:

| Script | Purpose |
|--------|---------|
| `scripts/check_online_status.py` | HTTP status checks for domains |
| `scripts/http_content_check.py` | Content-based online verification |
| `scripts/asn_classifier.py` | ASN/hosting classification |

## Configuration

Environment variables (see `.env.example`):

```bash
# Optional: API keys for extended features
SIMILARWEB_API_KEY=your_key_here
```

## Deployment

### Streamlit Cloud

The app is deployed on Streamlit Cloud and auto-deploys from the `main` branch.

To deploy your own instance:
1. Fork this repo
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Point to `app.py` as the main file

### Requirements

- Python 3.10+
- Dependencies in `requirements.txt`

## Domain Status Classification

| Status | Description |
|--------|-------------|
| 🟢 **Active** | High recent activity (last 90 days) |
| 🟡 **Low Activity** | Some recent activity |
| 🟠 **Declining** | Activity decreasing |
| ⚫ **Inactive** | No recent activity (>1 year) |
| ⚪ **Unknown** | Insufficient data |

## Project Structure

```
GTR-based-ranking/
├── app.py              # Main Streamlit application
├── config.py           # Configuration and constants
├── domain_checker.py   # Domain validation utilities
├── requirements.txt    # Python dependencies
├── run.sh              # Quick start script
├── data/               # Coverage and status data
│   ├── Coverage_Sites.xlsx
│   ├── VT_LSR_Site.xlsx
│   ├── V1_Sites.xlsx
│   └── online_status.csv
├── scripts/            # Utility scripts
│   ├── check_online_status.py
│   ├── http_content_check.py
│   └── asn_classifier.py
└── sample_data.parquet # GTR domain dataset
```

## Contributing

1. Create a feature branch
2. Make changes
3. Test locally with `streamlit run app.py`
4. Submit a pull request

## License

Internal Vobile use only.
