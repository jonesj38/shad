"""Content ingestion for various source types."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of an ingestion operation."""

    success: bool
    files_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class URLIngester:
    """Ingest content from web URLs."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path
        self._sources_dir = vault_path / "Sources"
        self._sources_dir.mkdir(exist_ok=True)

    async def ingest(self, url: str) -> IngestResult:
        """Fetch and convert a URL to markdown."""
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
                content_type = response.headers.get("content-type", "")
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return IngestResult(success=False, errors=[str(e)])

        # Convert HTML to markdown
        if "text/html" in content_type:
            markdown = self._html_to_markdown(html, url)
        elif "text/plain" in content_type or "text/markdown" in content_type:
            markdown = html
        else:
            markdown = f"```\n{html[:10000]}\n```"

        # Extract title
        title = self._extract_title(html) or self._url_to_title(url)

        # Generate file path
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        safe_title = re.sub(r"[^\w\s-]", "", title)[:50].strip().replace(" ", "-")

        snapshot_dir = self._sources_dir / domain / date_str
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        file_path = snapshot_dir / f"{safe_title}.md"

        # Create frontmatter
        content_hash = hashlib.sha256(markdown.encode()).hexdigest()[:16]
        frontmatter = f"""---
source_url: {url}
source_type: url
ingested_at: {datetime.now(UTC).isoformat()}
content_hash: sha256:{content_hash}
title: {title}
---

# {title}

"""

        # Write file
        file_path.write_text(frontmatter + markdown)

        return IngestResult(
            success=True,
            files_created=[str(file_path.relative_to(self.vault_path))],
            metadata={"title": title, "url": url, "hash": content_hash},
        )

    def _html_to_markdown(self, html: str, url: str) -> str:
        """Convert HTML to markdown."""
        try:
            # Try trafilatura first (best for articles)
            import trafilatura
            result = trafilatura.extract(
                html,
                include_links=True,
                include_formatting=True,
                include_tables=True,
                url=url,
            )
            if result:
                return result
        except ImportError:
            pass

        try:
            # Fallback to markdownify
            from markdownify import markdownify
            return markdownify(html, heading_style="ATX", strip=["script", "style"])
        except ImportError:
            pass

        # Last resort: strip HTML tags
        clean = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()[:10000]

    def _extract_title(self, html: str) -> str | None:
        """Extract title from HTML."""
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _url_to_title(self, url: str) -> str:
        """Generate title from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if path:
            return path.split("/")[-1].replace("-", " ").replace("_", " ").title()
        return parsed.netloc


class FeedIngester:
    """Ingest content from RSS/Atom feeds."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path
        self._sources_dir = vault_path / "Sources"
        self._sources_dir.mkdir(exist_ok=True)

    async def ingest(self, feed_url: str, max_items: int = 10) -> IngestResult:
        """Fetch and process a feed."""
        try:
            import feedparser
        except ImportError:
            return IngestResult(
                success=False,
                errors=["feedparser not installed. Run: pip install feedparser"],
            )

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                response = await client.get(feed_url)
                response.raise_for_status()
                feed_content = response.text
        except Exception as e:
            logger.error(f"Failed to fetch feed {feed_url}: {e}")
            return IngestResult(success=False, errors=[str(e)])

        feed = feedparser.parse(feed_content)

        if feed.bozo and not feed.entries:
            return IngestResult(
                success=False,
                errors=[f"Failed to parse feed: {feed.bozo_exception}"],
            )

        # Create feed directory
        parsed = urlparse(feed_url)
        domain = parsed.netloc.replace("www.", "")
        feed_title = feed.feed.get("title", domain)
        safe_feed_title = re.sub(r"[^\w\s-]", "", feed_title)[:30].strip().replace(" ", "-")

        feed_dir = self._sources_dir / "feeds" / safe_feed_title
        feed_dir.mkdir(parents=True, exist_ok=True)

        files_created = []
        errors = []

        for entry in feed.entries[:max_items]:
            try:
                file_path = self._process_entry(entry, feed_dir, feed_url)
                if file_path:
                    files_created.append(str(file_path.relative_to(self.vault_path)))
            except Exception as e:
                errors.append(f"Failed to process entry: {e}")

        return IngestResult(
            success=len(files_created) > 0,
            files_created=files_created,
            errors=errors,
            metadata={"feed_title": feed_title, "entries_processed": len(files_created)},
        )

    def _process_entry(self, entry: Any, feed_dir: Path, feed_url: str) -> Path | None:
        """Process a single feed entry."""
        title = entry.get("title", "Untitled")
        link = entry.get("link", "")
        published = entry.get("published", "")
        summary = entry.get("summary", "")
        content = ""

        # Get full content if available
        if "content" in entry:
            content = entry.content[0].get("value", "")

        # Use summary if no content
        if not content:
            content = summary

        # Clean HTML from content
        content = self._clean_html(content)

        # Generate filename
        safe_title = re.sub(r"[^\w\s-]", "", title)[:50].strip().replace(" ", "-")
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        file_path = feed_dir / f"{date_str}-{safe_title}.md"

        # Skip if already exists
        if file_path.exists():
            return None

        # Create frontmatter
        frontmatter = f"""---
source_url: {link}
source_type: feed
feed_url: {feed_url}
ingested_at: {datetime.now(UTC).isoformat()}
published: {published}
title: {title}
---

# {title}

"""

        if link:
            frontmatter += f"Source: [{link}]({link})\n\n"

        file_path.write_text(frontmatter + content)
        return file_path

    def _clean_html(self, html: str) -> str:
        """Strip HTML tags from content."""
        clean = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()


class FolderIngester:
    """Watch and sync local folders."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path
        self._sources_dir = vault_path / "Sources"
        self._sources_dir.mkdir(exist_ok=True)

    async def ingest(self, folder_path: str) -> IngestResult:
        """Sync files from a local folder."""
        source_path = Path(folder_path).expanduser().resolve()

        if not source_path.exists():
            return IngestResult(
                success=False,
                errors=[f"Folder not found: {folder_path}"],
            )

        folder_name = source_path.name
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        target_dir = self._sources_dir / "local" / folder_name / date_str
        target_dir.mkdir(parents=True, exist_ok=True)

        files_created = []
        errors = []

        # Copy markdown and text files
        for file_path in source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in (".md", ".txt", ".rst"):
                try:
                    rel_path = file_path.relative_to(source_path)
                    target_path = target_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy with frontmatter
                    content = file_path.read_text()
                    if not content.startswith("---"):
                        frontmatter = f"""---
source_path: {file_path}
source_type: folder
ingested_at: {datetime.now(UTC).isoformat()}
---

"""
                        content = frontmatter + content

                    target_path.write_text(content)
                    files_created.append(str(target_path.relative_to(self.vault_path)))
                except Exception as e:
                    errors.append(f"Failed to copy {file_path}: {e}")

        return IngestResult(
            success=len(files_created) > 0,
            files_created=files_created,
            errors=errors,
            metadata={"source_folder": str(source_path), "files_copied": len(files_created)},
        )
