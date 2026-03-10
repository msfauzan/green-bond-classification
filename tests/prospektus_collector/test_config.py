import os
import importlib
import pytest
from unittest.mock import patch


def test_config_reads_firecrawl_api_key():
    with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}):
        import prospektus_collector.config as cfg
        importlib.reload(cfg)
        assert cfg.FIRECRAWL_API_KEY == "test-key"


def test_config_reads_r2_settings():
    env = {
        "R2_BUCKET_NAME": "my-bucket",
        "R2_ACCOUNT_ID": "acc123",
        "R2_ACCESS_KEY_ID": "key123",
        "R2_SECRET_ACCESS_KEY": "secret123",
    }
    with patch.dict(os.environ, env):
        import prospektus_collector.config as cfg
        importlib.reload(cfg)
        assert cfg.R2_BUCKET_NAME == "my-bucket"
        assert cfg.R2_ACCOUNT_ID == "acc123"


def test_config_pojk_date_default():
    env = {k: v for k, v in os.environ.items() if k != "POJK_18_DATE"}
    with patch.dict(os.environ, env, clear=True):
        import prospektus_collector.config as cfg
        importlib.reload(cfg)
        assert cfg.POJK_18_DATE == "2023-01-01"


def test_config_pdf_keywords_is_list():
    import prospektus_collector.config as cfg
    importlib.reload(cfg)
    assert isinstance(cfg.PDF_KEYWORDS, list)
    assert "prospektus" in cfg.PDF_KEYWORDS


def test_config_r2_endpoint_url_derived():
    with patch.dict(os.environ, {"R2_ACCOUNT_ID": "myaccount123"}):
        import prospektus_collector.config as cfg
        importlib.reload(cfg)
        assert "myaccount123" in cfg.R2_ENDPOINT_URL
        assert "r2.cloudflarestorage.com" in cfg.R2_ENDPOINT_URL
