"""Content chunking utilities for crawled content."""

import re
from typing import Any

from pydantic import BaseModel


class Chunk(BaseModel):
    """A chunk of content with metadata."""

    text: str
    index: int
    headers: str
    char_count: int
    word_count: int


def split_by_header(md: str, header_pattern: str) -> list[str]:
    """Split markdown by a specific header pattern."""
    indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
    indices.append(len(md))
    return [
        md[indices[i] : indices[i + 1]].strip()
        for i in range(len(indices) - 1)
        if md[indices[i] : indices[i + 1]].strip()
    ]


def _chunk_recursive(sections: list[str], pattern: str, max_len: int) -> list[str]:
    """Recursively split sections by sub-headers."""
    result: list[str] = []
    for section in sections:
        if len(section) > max_len:
            sub_pattern = r"^#{}\s.+$".format(pattern.count("#") + 1)
            if sub_pattern.count("#") <= 3:
                subs = split_by_header(section, sub_pattern)
                if len(subs) > 1:
                    result.extend(_chunk_recursive(subs, sub_pattern, max_len))
                    continue
            result.extend(section[i : i + max_len].strip() for i in range(0, len(section), max_len))
        else:
            result.append(section)
    return result


def smart_chunk_markdown(markdown: str, max_len: int = 1000) -> list[Chunk]:
    """
    Hierarchically split markdown by #, ##, ### headers, then by characters.

    Ensures all chunks are less than max_len while preserving header context.

    Args:
        markdown: Markdown content to chunk
        max_len: Maximum characters per chunk

    Returns:
        List of Chunk objects
    """
    chunks = _chunk_recursive(split_by_header(markdown, r"^# .+$"), r"^# .+$", max_len)

    result_chunks = []
    for idx, text in enumerate([c for c in chunks if c]):
        headers = extract_headers(text)
        result_chunks.append(
            Chunk(
                text=text,
                index=idx,
                headers=headers,
                char_count=len(text),
                word_count=len(text.split()),
            )
        )

    return result_chunks


def extract_headers(chunk: str) -> str:
    """Extract headers from a chunk for context."""
    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    return "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""


def truncate_content(content: str, max_length: int = 100000) -> str:
    """Truncate content to maximum length with warning."""
    if len(content) <= max_length:
        return content

    truncated = content[:max_length]
    truncated += (
        f"\n\n[Content truncated at {max_length} characters. Full content available upon request.]"
    )
    return truncated


def extract_title_from_markdown(markdown: str) -> str | None:
    """Extract title from markdown content."""
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def clean_markdown(markdown: str) -> str:
    """Clean and normalize markdown content."""
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r" {2,}", " ", markdown)
    return markdown.strip()


def get_chunk_summary(chunk: Chunk) -> dict[str, Any]:
    """Get a summary of a chunk."""
    return {
        "index": chunk.index,
        "headers": chunk.headers,
        "char_count": chunk.char_count,
        "word_count": chunk.word_count,
        "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
    }
