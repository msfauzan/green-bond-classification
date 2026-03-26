"""Configuration for Prospektus Collector."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
ML_DATASET_DIR = BASE_DIR / "ML_Dataset"

# Firecrawl
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_TIMEOUT = int(os.getenv("FIRECRAWL_TIMEOUT", "30"))

# Cloudflare R2
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_R2_BUCKET = os.getenv("CLOUDFLARE_R2_BUCKET", "prospektus")
CLOUDFLARE_R2_ENDPOINT = os.getenv("CLOUDFLARE_R2_ENDPOINT", "")
CLOUDFLARE_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_ACCESS_KEY_ID", "")
CLOUDFLARE_SECRET_ACCESS_KEY = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY", "")

# Collection settings
POJK_18_DATE = "2023-01-01"
PDF_KEYWORDS = [
    "prospektus", "penawaran umum", "efek bersifat utang",
    "obligasi", "sukuk", "eba", "medium term notes", "mtn"
]
MAX_CONCURRENT_REQUESTS = 5
