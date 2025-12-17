#!/usr/bin/env python3
"""Fetch and clean Python documentation for training.

This script downloads Python official documentation and processes it
into clean text suitable for language model training.

Usage:
    python scripts/fetch_python_docs.py --output data/python_docs
    python scripts/fetch_python_docs.py --output data/python_docs --max-files 500
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_html_to_text(html: str) -> str:
    """Extract clean text from HTML content.

    Args:
        html: Raw HTML string

    Returns:
        Cleaned text content
    """
    import html as html_module

    # Remove script and style elements
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Replace br and p tags with newlines BEFORE removing tags
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</p>', '\n\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</div>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</li>', '\n', html, flags=re.IGNORECASE)

    # Remove all remaining HTML tags BEFORE decoding entities
    # This prevents decoded < and > from being treated as tags
    html = re.sub(r'<[^>]+>', '', html)

    # NOW decode HTML entities (after tags are removed)
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&amp;', '&')
    html = html.replace('&quot;', '"')
    html = html.replace('&#39;', "'")
    html = html.replace('&#60;', '<')
    html = html.replace('&#62;', '>')

    # Use html.unescape for any remaining entities
    html = html_module.unescape(html)

    # Clean up whitespace
    lines = []
    for line in html.split('\n'):
        line = line.strip()
        if line:
            # Normalize whitespace within lines
            line = re.sub(r'\s+', ' ', line)
            lines.append(line)

    text = '\n'.join(lines)

    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_code_blocks(text: str) -> str:
    """Format code blocks for training.

    Args:
        text: Text that may contain code

    Returns:
        Text with code blocks formatted
    """
    # Already extracted - just ensure proper formatting
    return text


def clean_python_doc(text: str) -> str:
    """Clean Python documentation text.

    Args:
        text: Raw documentation text

    Returns:
        Cleaned text suitable for training
    """
    # Remove navigation elements (common patterns in Python docs)
    text = re.sub(r'previous\s*\|\s*next\s*\|', '', text, flags=re.IGNORECASE)
    text = re.sub(r'index\s*\|\s*modules', '', text, flags=re.IGNORECASE)

    # Remove version info clutter
    text = re.sub(r'New in version \d+\.\d+\.?', '', text)
    text = re.sub(r'Changed in version \d+\.\d+\.?', '', text)
    text = re.sub(r'Deprecated since version \d+\.\d+\.?', '', text)

    # Clean up
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def fetch_url(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch content from URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Response text or None on failure
    """
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; PythonDocsBot/1.0)'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode('utf-8', errors='replace')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def get_python_doc_urls() -> list:
    """Get list of Python documentation URLs to fetch.

    Returns:
        List of documentation page URLs
    """
    base_url = "https://docs.python.org/3/"

    # Core documentation sections
    sections = [
        # Tutorial
        "tutorial/index.html",
        "tutorial/appetite.html",
        "tutorial/interpreter.html",
        "tutorial/introduction.html",
        "tutorial/controlflow.html",
        "tutorial/datastructures.html",
        "tutorial/modules.html",
        "tutorial/inputoutput.html",
        "tutorial/errors.html",
        "tutorial/classes.html",
        "tutorial/stdlib.html",
        "tutorial/stdlib2.html",
        "tutorial/venv.html",

        # Library Reference (most important modules)
        "library/functions.html",
        "library/stdtypes.html",
        "library/exceptions.html",
        "library/string.html",
        "library/re.html",
        "library/collections.html",
        "library/itertools.html",
        "library/functools.html",
        "library/operator.html",
        "library/pathlib.html",
        "library/os.html",
        "library/os.path.html",
        "library/shutil.html",
        "library/glob.html",
        "library/io.html",
        "library/time.html",
        "library/datetime.html",
        "library/calendar.html",
        "library/math.html",
        "library/random.html",
        "library/statistics.html",
        "library/json.html",
        "library/csv.html",
        "library/sqlite3.html",
        "library/typing.html",
        "library/dataclasses.html",
        "library/contextlib.html",
        "library/abc.html",
        "library/copy.html",
        "library/pprint.html",
        "library/enum.html",
        "library/logging.html",
        "library/argparse.html",
        "library/unittest.html",
        "library/doctest.html",
        "library/sys.html",
        "library/traceback.html",
        "library/gc.html",
        "library/inspect.html",
        "library/dis.html",
        "library/pickle.html",
        "library/shelve.html",
        "library/marshal.html",
        "library/dbm.html",
        "library/subprocess.html",
        "library/multiprocessing.html",
        "library/threading.html",
        "library/concurrent.futures.html",
        "library/asyncio.html",
        "library/socket.html",
        "library/ssl.html",
        "library/select.html",
        "library/email.html",
        "library/html.html",
        "library/xml.html",
        "library/urllib.html",
        "library/http.html",
        "library/ftplib.html",
        "library/smtplib.html",
        "library/struct.html",
        "library/codecs.html",
        "library/hashlib.html",
        "library/hmac.html",
        "library/secrets.html",
        "library/tempfile.html",
        "library/zipfile.html",
        "library/tarfile.html",
        "library/gzip.html",
        "library/bz2.html",
        "library/lzma.html",
        "library/zlib.html",

        # HOWTOs
        "howto/functional.html",
        "howto/logging.html",
        "howto/regex.html",
        "howto/sorting.html",
        "howto/unicode.html",
        "howto/urllib2.html",
        "howto/argparse.html",
        "howto/descriptor.html",
        "howto/enum.html",

        # Language Reference
        "reference/datamodel.html",
        "reference/executionmodel.html",
        "reference/import.html",
        "reference/expressions.html",
        "reference/simple_stmts.html",
        "reference/compound_stmts.html",

        # FAQ
        "faq/general.html",
        "faq/programming.html",
        "faq/design.html",
        "faq/library.html",
    ]

    return [urljoin(base_url, section) for section in sections]


def fetch_python_docs(
    output_dir: str,
    max_files: Optional[int] = None,
    min_length: int = 500,
) -> dict:
    """Fetch and process Python documentation.

    Args:
        output_dir: Directory to save processed files
        max_files: Maximum number of files to fetch (None for all)
        min_length: Minimum text length to keep a file

    Returns:
        Statistics about fetched content
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    urls = get_python_doc_urls()
    if max_files:
        urls = urls[:max_files]

    stats = {
        "total_urls": len(urls),
        "successful": 0,
        "failed": 0,
        "too_short": 0,
        "total_chars": 0,
        "total_files": 0,
    }

    logger.info(f"Fetching {len(urls)} Python documentation pages...")

    for i, url in enumerate(urls):
        # Create filename from URL
        parsed = urlparse(url)
        filename = parsed.path.replace('/', '_').replace('.html', '.txt')
        filename = filename.lstrip('_')
        if not filename:
            filename = f"page_{i}.txt"

        # Fetch content
        logger.info(f"[{i+1}/{len(urls)}] Fetching {url}")
        html = fetch_url(url)

        if html is None:
            stats["failed"] += 1
            continue

        # Process content
        text = clean_html_to_text(html)
        text = clean_python_doc(text)

        # Check minimum length
        if len(text) < min_length:
            stats["too_short"] += 1
            logger.debug(f"Skipping {filename}: too short ({len(text)} chars)")
            continue

        # Save file
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        stats["successful"] += 1
        stats["total_chars"] += len(text)
        stats["total_files"] += 1

        logger.debug(f"Saved {filename} ({len(text)} chars)")

    return stats


def create_combined_dataset(
    input_dir: str,
    output_file: str,
    separator: str = "\n\n---\n\n",
) -> dict:
    """Combine all text files into a single training file.

    Args:
        input_dir: Directory containing text files
        output_file: Path to output combined file
        separator: Separator between documents

    Returns:
        Statistics about combined dataset
    """
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.txt"))

    stats = {
        "files": len(files),
        "total_chars": 0,
        "estimated_tokens": 0,
    }

    with open(output_file, 'w', encoding='utf-8') as out:
        for i, file_path in enumerate(files):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if i > 0:
                out.write(separator)
            out.write(content)

            stats["total_chars"] += len(content)

    # Rough token estimate (4 chars per token for code/docs)
    stats["estimated_tokens"] = stats["total_chars"] // 4

    logger.info(f"Combined {stats['files']} files into {output_file}")
    logger.info(f"Total characters: {stats['total_chars']:,}")
    logger.info(f"Estimated tokens: {stats['estimated_tokens']:,}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and process Python documentation for LLM training"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/python_docs",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to fetch"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=500,
        help="Minimum text length to keep a file"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also create a combined single-file dataset"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("Python Documentation Fetcher")
    print("=" * 60)

    # Fetch documentation
    stats = fetch_python_docs(
        output_dir=args.output,
        max_files=args.max_files,
        min_length=args.min_length,
    )

    print("\nFetch Results:")
    print(f"  Total URLs: {stats['total_urls']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Too short: {stats['too_short']}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Est. tokens: {stats['total_chars'] // 4:,}")

    # Optionally combine into single file
    if args.combine:
        combined_file = f"{args.output}_combined.txt"
        combine_stats = create_combined_dataset(args.output, combined_file)
        print(f"\nCombined file: {combined_file}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
