"""Crawler feature constants."""

from enum import Enum


class CrawlMode(str, Enum):
    """Crawl output modes."""

    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    SUMMARY = "summary"


class SchemaType(str, Enum):
    """Predefined schema types."""

    PRODUCT = "product"
    ARTICLE = "article"
    PERSON = "person"
    JOB = "job"
    CUSTOM = "custom"


CRAWLER_TAG = "Crawler"
CRAWLER_PREFIX = "/crawler"
