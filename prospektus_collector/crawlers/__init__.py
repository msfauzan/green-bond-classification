"""Crawlers for collecting prospektus from various sources."""
from .base_crawler import BaseCrawler
from .idx_crawler import IDXCrawler

__all__ = ["BaseCrawler", "IDXCrawler"]
