#!/usr/bin/env python3
"""Clean training data for language model.

This script cleans text data by:
- Removing HTML entities
- Removing navigation boilerplate
- Removing repeated headers/footers
- Cleaning up whitespace
- Filtering low-quality lines

Usage:
    python scripts/clean_data.py data/python_docs_full -o data/python_docs_clean
"""

import argparse
import html
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Patterns to remove entirely
REMOVE_PATTERNS = [
    # Navigation elements
    r'^Theme\s*$',
    r'^Auto\s*$',
    r'^Light\s*$',
    r'^Dark\s*$',
    r'^Previous topic\s*$',
    r'^Next topic\s*$',
    r'^This page\s*$',
    r'^Report a bug\s*$',
    r'^Show source\s*$',
    r'^Navigation\s*$',
    r'^index\s*$',
    r'^modules\s*\|?\s*$',
    r'^next\s*\|?\s*$',
    r'^previous\s*\|?\s*$',
    r'^Table of Contents\s*$',

    # Headers/footers
    r'^Python\s*»\s*$',
    r'^Python 3\.\d+\.\d+ Documentation\s*»?\s*$',
    r'^\d+\.\d+\.\d+ Documentation\s*$',
    r'^The Python Standard Library\s*»?\s*$',
    r'^The Python Tutorial\s*»?\s*$',

    # Version markers
    r'^New in version \d+\.\d+',
    r'^Changed in version \d+\.\d+',
    r'^Deprecated since version \d+\.\d+',

    # Source code references
    r'^Source code: Lib/',

    # Pipe separators
    r'^\s*\|\s*$',

    # Single special characters
    r'^[»|¶]+\s*$',
]

# Compiled patterns for efficiency
REMOVE_REGEX = [re.compile(p, re.IGNORECASE) for p in REMOVE_PATTERNS]

# HTML entities to decode
HTML_ENTITIES = {
    '&#8212;': '—',
    '&#8211;': '–',
    '&#8217;': "'",
    '&#8216;': "'",
    '&#8220;': '"',
    '&#8221;': '"',
    '&#8230;': '...',
    '&#187;': '',  # Remove these navigation arrows
    '&#171;': '',
    '&#64;': '@',
    '&#39;': "'",
    '&#34;': '"',
    '&#x27;': "'",
    '&#x22;': '"',
    '&#60;': '<',
    '&#62;': '>',
    '&#x3c;': '<',
    '&#x3e;': '>',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&nbsp;': ' ',
    '&quot;': '"',
    '&apos;': "'",
    '&mdash;': '—',
    '&ndash;': '–',
    '&hellip;': '...',
    '&copy;': '©',
    '&reg;': '®',
    '&trade;': '™',
    '&para;': '',  # Remove pilcrow
    '&sect;': '§',
}


def clean_html_entities(text: str) -> str:
    """Replace HTML entities with their character equivalents."""
    # First, handle named entities
    for entity, replacement in HTML_ENTITIES.items():
        text = text.replace(entity, replacement)

    # Then handle any remaining numeric entities
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)

    # Final pass with html.unescape for anything we missed
    text = html.unescape(text)

    return text


def should_remove_line(line: str) -> bool:
    """Check if a line should be removed."""
    stripped = line.strip()

    # Empty lines are fine
    if not stripped:
        return False

    # Check against removal patterns
    for pattern in REMOVE_REGEX:
        if pattern.match(stripped):
            return True

    # Remove very short lines that are likely navigation
    if len(stripped) < 3 and not stripped.isalnum():
        return True

    # Remove lines that are just punctuation/symbols
    if re.match(r'^[\|\-\*\#\=\>\<\¶]+$', stripped):
        return True

    return False


def is_toc_line(line: str) -> bool:
    """Check if line looks like table of contents entry."""
    stripped = line.strip()

    # Numbered section references like "4.9.3.1. Positional-or-Keyword"
    if re.match(r'^\d+(\.\d+)+\.?\s+\w', stripped):
        # But keep if it's actual content (has enough text)
        if len(stripped) > 60:
            return False
        return True

    return False


def clean_line(line: str) -> str:
    """Clean a single line of text."""
    # Remove HTML entities
    line = clean_html_entities(line)

    # Remove pilcrow (¶) characters often used as section markers
    line = line.replace('¶', '')

    # Normalize whitespace
    line = re.sub(r'\s+', ' ', line)

    # Remove trailing/leading whitespace
    line = line.strip()

    return line


def is_boilerplate_line(line: str, seen_lines: set, in_header: bool = False) -> bool:
    """Check if line is boilerplate that should be removed."""
    stripped = line.strip()

    # Skip empty
    if not stripped:
        return False

    # Duplicate line (exact match)
    if stripped in seen_lines and len(stripped) < 100:
        return True

    # Title with version suffix
    if '— Python 3.' in stripped and 'documentation' in stripped:
        return True

    # Cross-references to other modules (common in headers)
    if re.match(r'^[\w\.]+ — .+$', stripped) and len(stripped) < 80:
        if in_header:
            return True

    # Section references without content (in headers)
    if in_header and re.match(r'^[\w\s\.\(\)\-\'\"]+$', stripped) and len(stripped) < 60:
        # Looks like a TOC entry or navigation
        words = stripped.split()
        if len(words) <= 6 and not any(c in stripped for c in ',:;?!'):
            return True

    # Single word lines that are likely navigation
    if len(stripped.split()) == 1 and len(stripped) < 20:
        if stripped.lower() in {'python', 'changelog', 'tip', 'note', 'warning',
                                 'see', 'also', 'contents', 'examples', 'seealso'}:
            return True

    # Numbered section references like "3. Something" or "4.1. Something"
    if in_header and re.match(r'^\d+(\.\d+)*\.?\s+\w', stripped) and len(stripped) < 60:
        return True

    return False


def clean_file(input_path: Path) -> Tuple[str, dict]:
    """Clean a single file.

    Returns:
        Tuple of (cleaned_text, stats_dict)
    """
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.split('\n')
    original_lines = len(lines)

    # First pass: clean all lines
    cleaned_lines_pass1 = []
    for line in lines:
        cleaned = clean_line(line)
        if cleaned:
            cleaned_lines_pass1.append(cleaned)

    # Second pass: remove boilerplate and duplicates
    seen_lines: set = set()
    cleaned_lines = []
    removed_count = 0
    toc_count = 0

    # Find where actual content starts (skip header boilerplate)
    content_start = 0
    for i, line in enumerate(cleaned_lines_pass1):
        # Look for markers that indicate real content
        if any(marker in line.lower() for marker in
               ['this module', 'this function', 'this class', 'this method',
                'provides', 'allows', 'returns', 'takes', 'implements']):
            content_start = i
            break
        if i > 40:  # Give up after 40 lines
            content_start = min(20, len(cleaned_lines_pass1))  # Skip first 20
            break

    for i, line in enumerate(cleaned_lines_pass1):
        # Be more aggressive with early lines
        in_header = i < content_start

        # Skip lines that match removal patterns
        if should_remove_line(line):
            removed_count += 1
            continue

        # Skip TOC-like lines
        if is_toc_line(line):
            if in_header or len(line) < 60:
                toc_count += 1
                continue

        # Skip boilerplate
        if is_boilerplate_line(line, seen_lines, in_header):
            removed_count += 1
            continue

        # Track seen lines for duplicate detection
        if len(line) < 100:
            seen_lines.add(line)

        cleaned_lines.append(line)

    # Join lines and clean up
    text = '\n'.join(cleaned_lines)

    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just the module/function name repeated
    lines_final = text.split('\n')
    if len(lines_final) > 2:
        # If first line is just a title, and it's repeated, remove duplicates
        first_line = lines_final[0].strip()
        text = '\n'.join(
            line for i, line in enumerate(lines_final)
            if i == 0 or line.strip() != first_line
        )

    stats = {
        'original_lines': original_lines,
        'cleaned_lines': len(cleaned_lines),
        'removed_lines': removed_count,
        'toc_lines': toc_count,
        'original_chars': len(content),
        'cleaned_chars': len(text),
    }

    return text.strip(), stats


def clean_directory(
    input_dir: str,
    output_dir: str,
    min_length: int = 200,
) -> dict:
    """Clean all text files in a directory.

    Args:
        input_dir: Input directory with text files
        output_dir: Output directory for cleaned files
        min_length: Minimum character length to keep a file

    Returns:
        Statistics dictionary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.txt"))
    logger.info(f"Found {len(files)} files to clean")

    total_stats = {
        'files_processed': 0,
        'files_kept': 0,
        'files_skipped': 0,
        'original_chars': 0,
        'cleaned_chars': 0,
        'lines_removed': 0,
    }

    for file_path in sorted(files):
        try:
            cleaned_text, stats = clean_file(file_path)

            total_stats['files_processed'] += 1
            total_stats['original_chars'] += stats['original_chars']
            total_stats['cleaned_chars'] += len(cleaned_text)
            total_stats['lines_removed'] += stats['removed_lines'] + stats['toc_lines']

            # Skip files that are too short after cleaning
            if len(cleaned_text) < min_length:
                total_stats['files_skipped'] += 1
                logger.debug(f"Skipping {file_path.name}: too short ({len(cleaned_text)} chars)")
                continue

            # Write cleaned file
            output_file = output_path / file_path.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            total_stats['files_kept'] += 1

            reduction = (1 - len(cleaned_text) / stats['original_chars']) * 100
            logger.debug(f"Cleaned {file_path.name}: {reduction:.1f}% reduction")

        except Exception as e:
            logger.error(f"Failed to clean {file_path}: {e}")

    return total_stats


def create_combined_file(
    input_dir: str,
    output_file: str,
    separator: str = "\n\n",
) -> dict:
    """Combine all cleaned files into a single training file."""
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.txt"))

    total_chars = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        for i, file_path in enumerate(files):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if i > 0:
                out.write(separator)
            out.write(content)
            total_chars += len(content)

    return {
        'files': len(files),
        'total_chars': total_chars,
        'estimated_tokens': total_chars // 4,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean training data for language model"
    )
    parser.add_argument(
        "input",
        help="Input directory with text files"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for cleaned files (default: input_clean)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum character length to keep a file"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also create a combined single-file dataset"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set output directory
    if args.output is None:
        args.output = args.input.rstrip('/') + '_clean'

    print("=" * 60)
    print("Data Cleaning Pipeline")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Clean files
    stats = clean_directory(args.input, args.output, args.min_length)

    reduction = (1 - stats['cleaned_chars'] / max(stats['original_chars'], 1)) * 100

    print(f"\nResults:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files kept: {stats['files_kept']}")
    print(f"  Files skipped (too short): {stats['files_skipped']}")
    print(f"  Lines removed: {stats['lines_removed']:,}")
    print(f"  Original size: {stats['original_chars']:,} chars")
    print(f"  Cleaned size: {stats['cleaned_chars']:,} chars")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Est. tokens: {stats['cleaned_chars'] // 4:,}")

    # Optionally combine
    if args.combine:
        combined_file = args.output + "_combined.txt"
        combine_stats = create_combined_file(args.output, combined_file)
        print(f"\nCombined file: {combined_file}")
        print(f"  Total chars: {combine_stats['total_chars']:,}")
        print(f"  Est. tokens: {combine_stats['estimated_tokens']:,}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
