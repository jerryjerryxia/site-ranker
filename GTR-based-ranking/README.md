# DACIS - Domain Activity & Coverage Intelligence System

GTR-based domain intelligence dashboard for analyzing piracy domains, their activity status, and coverage mapping.

## Overview

DACIS transforms Google Transparency Report (GTR) data from a passive archive into an **active intelligence asset**. It provides:

- **Notice Intelligence**: Lifetime notice counts, temporal history, trends
- **Operational Status**: Classify domains as Active/Inactive/Declining/Unknown
- **Coverage Mapping**: Track which domains are covered by our services
- **Ad-hoc Analysis**: CSV upload for quick domain list analysis

## Key Concept: What "Online" Really Means

A domain is considered **actively online** only if it shows signals consistent with ongoing piracy workflows:
- Recent enforcement activity (last 90 days)
- High notice velocity
- Consistent reporting patterns

The system classifies domains as:
- 🔴 **Active Piracy** - High recent activity
- 🟡 **Low Activity** - Some recent activity
- 🟠 **Declining** - Activity decreasing
- ⚫ **Inactive** - No recent activity (>1 year)
- ⚪ **Unknown** - Insufficient data

## Quick Start

```bash
# Option 1: Use the run script
./GTR-based-ranking/run.sh

# Option 2: Manual
source venv/bin/activate
cd GTR-based-ranking
streamlit run app.py
```

The dashboard will open at http://localhost:8501

## Dataset Options

- **Default (video_piracy_clean)**: ~50K curated video piracy domains - fast loading
- **Full GTR Dataset**: 6M+ domains - toggle in sidebar when needed

## Features

### 📊 Overview Tab
- Key metrics at a glance
- Status distribution charts
- Trend analysis

### 📋 Domain List Tab
- Full domain listing with pagination
- Sort by any column
- Export filtered data to CSV

### 📤 CSV Upload Tab
- Upload a CSV or TXT with domain list
- Automatic domain cleaning (www.google.co.uk → google.co.uk)
- Instant status and coverage check
- Download enriched results

### 🔎 Domain Detail Tab
- Deep-dive into individual domains
- Full notice intelligence
- Infrastructure signals

## Filters

- **Search**: Find specific domains
- **Operational Status**: Active/Inactive/Declining/Unknown
- **Coverage Status**: Covered vs Not Covered
- **Trend**: Rising/Stable/Declining
- **Notice Volume**: Min/max URL thresholds
- **Enforcement Flags**: Major Org / Major Studio enforcement

## Data Sources

The dashboard reads from:
- `data/processed/google_transparency_enriched.parquet` - Main GTR data

## Future Enhancements

- [ ] Real-time DNS/hosting infrastructure checks
- [ ] ASN and reverse proxy detection
- [ ] Site type classification (linking/streaming/index)
- [ ] Language detection
- [ ] ACE/MPAA priority site integration
- [ ] VideoTracker/VobileOne coverage integration
- [ ] Scheduled data refresh

## Architecture

```
GTR-based-ranking/
├── app.py           # Main Streamlit dashboard
├── README.md        # This file
└── (future modules)
```
