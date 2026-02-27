"""Crawler feature constants."""

from enum import StrEnum


class CrawlMode(StrEnum):
    """Crawl output modes."""

    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    SUMMARY = "summary"


class SchemaType(StrEnum):
    """Predefined schema types."""

    PRODUCT = "product"
    ARTICLE = "article"
    PERSON = "person"
    JOB = "job"
    CUSTOM = "custom"


CRAWLER_TAG = "Crawler"
CRAWLER_PREFIX = "/crawler"
