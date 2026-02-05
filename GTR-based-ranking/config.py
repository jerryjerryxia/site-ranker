"""
DACIS Configuration
"""
from pathlib import Path

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Dataset options
DATASETS = {
    "video_piracy": {
        "path": DATA_DIR / "video_piracy_clean.parquet",
        "description": "Curated video piracy domains (~50K)",
        "default": True
    },
    "full_gtr": {
        "path": DATA_DIR / "google_transparency_enriched.parquet", 
        "description": "Complete GTR dataset (6M+ domains)",
        "default": False
    }
}

# Coverage data sources (future integration)
COVERAGE_SOURCES = {
    "videotracker": {
        "enabled": False,
        "path": None,  # TODO: Add path when available
        "description": "VideoTracker coverage data"
    },
    "vobileone": {
        "enabled": False,
        "path": None,  # TODO: Add path when available
        "description": "VobileOne coverage data"
    }
}

# Operational status thresholds
STATUS_THRESHOLDS = {
    "active_piracy": {
        "days_inactive_max": 90,
        "velocity_min": 10
    },
    "low_activity": {
        "days_inactive_max": 180,
        "velocity_min": 0
    },
    "inactive": {
        "days_inactive_min": 365
    }
}

# Notice volume tiers
NOTICE_TIERS = [
    (0, 1000, "Very Low (<1K)"),
    (1000, 10000, "Low (1K-10K)"),
    (10000, 100000, "Medium (10K-100K)"),
    (100000, 1000000, "High (100K-1M)"),
    (1000000, float('inf'), "Very High (>1M)")
]

# Priority sources (ACE/MPAA) - placeholder for future data
PRIORITY_SOURCES = {
    "ace": {
        "enabled": False,
        "path": None  # TODO: Add when priority list is available
    },
    "mpaa": {
        "enabled": False,
        "path": None
    }
}
