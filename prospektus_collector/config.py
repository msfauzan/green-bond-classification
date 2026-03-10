import os

FIRECRAWL_API_KEY: str = os.environ.get("FIRECRAWL_API_KEY", "")
POJK_18_DATE: str = os.environ.get("POJK_18_DATE", "2023-01-01")

R2_BUCKET_NAME: str = os.environ.get("R2_BUCKET_NAME", "green-bond-storage")
R2_ACCOUNT_ID: str = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID: str = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY: str = os.environ.get("R2_SECRET_ACCESS_KEY", "")
# Derived: https://{account_id}.r2.cloudflarestorage.com (atau override custom domain)
R2_ENDPOINT_URL: str = os.environ.get(
    "R2_ENDPOINT_URL",
    f"https://{os.environ.get('R2_ACCOUNT_ID', '')}.r2.cloudflarestorage.com",
)

MAX_CONCURRENT_EMITEN: int = int(os.environ.get("MAX_CONCURRENT_EMITEN", "10"))
FIRECRAWL_TIMEOUT: int = int(os.environ.get("FIRECRAWL_TIMEOUT", "30"))

PDF_KEYWORDS: list = [
    "prospektus",
    "penawaran umum",
    "efek bersifat utang",
    "obligasi",
    "sukuk",
    "eba",
    "medium term notes",
    "mtn",
]

EMITEN_IR_KEYWORDS: list = [
    "investor-relations",
    "investor_relations",
    "hubungan-investor",
    "hubungan_investor",
    "publikasi",
    "publications",
    "prospektus",
    "disclosure",
    "keterbukaan",
]
