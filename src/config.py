"""Configuration management for site-ranker.

Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class Config:
    """Application configuration."""

    # Telegram Bot API
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")

    # Rate limits (requests per second)
    TELEGRAM_RATE_LIMIT: float = float(os.getenv("TELEGRAM_RATE_LIMIT", "1.0"))
    SITE_SCRAPER_RATE_LIMIT: float = float(os.getenv("SITE_SCRAPER_RATE_LIMIT", "0.5"))

    # Timeouts (seconds)
    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "15"))

    # Data paths
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    RAW_DIR = DATA_DIR / "raw"
    INPUTS_DIR = DATA_DIR / "inputs"
    OUTPUTS_DIR = DATA_DIR / "outputs"

    # File paths
    TRANCO_APEX = PROJECT_ROOT / "tranco_08_12_2025.csv"
    TRANCO_SUB = PROJECT_ROOT / "tranco_08_12_2025_sub.csv"
    GTR_PARQUET = PROCESSED_DIR / "google_transparency_by_domain.parquet"

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of warnings/errors
        """
        warnings = []

        if not cls.TELEGRAM_BOT_TOKEN:
            warnings.append(
                "TELEGRAM_BOT_TOKEN not set - Telegram subscriber counts will be unavailable. "
                "Get a token from @BotFather on Telegram."
            )

        if not cls.TRANCO_APEX.exists():
            warnings.append(f"Tranco apex list not found: {cls.TRANCO_APEX}")

        if not cls.GTR_PARQUET.exists():
            warnings.append(
                f"Google Transparency data not found: {cls.GTR_PARQUET}. "
                "Run scripts/preprocess_google_transparency_chunked.py first."
            )

        return warnings

    @classmethod
    def print_status(cls):
        """Print configuration status."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Site Ranker Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Status", style="green")

        # Telegram
        if cls.TELEGRAM_BOT_TOKEN:
            token_preview = cls.TELEGRAM_BOT_TOKEN[:10] + "..." if len(cls.TELEGRAM_BOT_TOKEN) > 10 else "***"
            table.add_row("Telegram Bot Token", token_preview, "✓ Set")
        else:
            table.add_row("Telegram Bot Token", "Not set", "✗ Missing")

        # Rate limits
        table.add_row("Telegram Rate Limit", f"{cls.TELEGRAM_RATE_LIMIT} req/s", "✓")
        table.add_row("Site Scraper Rate Limit", f"{cls.SITE_SCRAPER_RATE_LIMIT} req/s", "✓")

        # Paths
        table.add_row(
            "Tranco Data",
            "Found" if cls.TRANCO_APEX.exists() else "Missing",
            "✓" if cls.TRANCO_APEX.exists() else "✗"
        )
        table.add_row(
            "GTR Data",
            "Found" if cls.GTR_PARQUET.exists() else "Missing",
            "✓" if cls.GTR_PARQUET.exists() else "✗"
        )

        console.print(table)

        # Print warnings
        warnings = cls.validate()
        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  ⚠️  {warning}")


# Singleton instance
config = Config()
