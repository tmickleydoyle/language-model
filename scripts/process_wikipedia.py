"""
Extract and process Simple Wikipedia XML dump into clean text files.

This script:
1. Decompresses the .bz2 file
2. Parses the MediaWiki XML format
3. Extracts article text and cleans markup
4. Saves articles as individual text files
"""

import bz2
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse


def clean_wiki_text(text: str) -> str:
    """Clean MediaWiki markup from article text."""
    # Remove XML/HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove wiki markup
    text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Templates
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)  # Categories
    text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)  # Files
    text = re.sub(r'\[\[Image:[^\]]+\]\]', '', text)  # Images

    # Convert wiki links [[link|text]] or [[link]]
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Remove external links
    text = re.sub(r'\[http[^\]]+\]', '', text)

    # Remove reference tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/?>', '', text)

    # Remove multiple newlines
    text = re.sub(r'\n\n+', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_articles(xml_path: Path, output_dir: Path, max_articles: int = None, min_length: int = 500):
    """Extract articles from Wikipedia XML dump.

    Args:
        xml_path: Path to the .bz2 compressed XML file
        output_dir: Directory to save extracted articles
        max_articles: Maximum number of articles to extract (None = all)
        min_length: Minimum article length in characters
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📖 Processing Wikipedia dump: {xml_path.name}")
    print(f"💾 Output directory: {output_dir}")

    # Open compressed file
    with bz2.open(xml_path, 'rt', encoding='utf-8') as f:
        article_count = 0
        total_chars = 0

        # Parse XML incrementally to handle large files
        context = ET.iterparse(f, events=('start', 'end'))
        context = iter(context)
        _, root = next(context)

        current_page = {}
        in_page = False

        for event, elem in context:
            tag = elem.tag.split('}')[-1]  # Remove namespace

            if event == 'start':
                if tag == 'page':
                    in_page = True
                    current_page = {}

            elif event == 'end':
                if tag == 'title' and in_page:
                    current_page['title'] = elem.text or ''

                elif tag == 'text' and in_page:
                    text = elem.text or ''
                    current_page['text'] = text

                elif tag == 'page' and in_page:
                    in_page = False

                    # Process the article
                    title = current_page.get('title', '')
                    text = current_page.get('text', '')

                    # Skip special pages
                    if ':' in title or not text:
                        elem.clear()
                        continue

                    # Clean the text
                    clean_text = clean_wiki_text(text)

                    # Skip short articles
                    if len(clean_text) < min_length:
                        elem.clear()
                        continue

                    # Save article
                    article_count += 1
                    total_chars += len(clean_text)

                    # Create safe filename
                    safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                    safe_title = re.sub(r'\s+', '_', safe_title)
                    filename = f"wiki_{article_count:06d}_{safe_title}.txt"

                    article_path = output_dir / filename
                    article_path.write_text(clean_text, encoding='utf-8')

                    if article_count % 100 == 0:
                        print(f"  ✅ Extracted {article_count} articles ({total_chars:,} chars)")

                    # Check if we've reached the limit
                    if max_articles and article_count >= max_articles:
                        print(f"\n✅ Reached limit of {max_articles} articles")
                        break

                # Clear element to save memory
                elem.clear()
                root.clear()

    print(f"\n🎉 Extraction complete!")
    print(f"   📄 Total articles: {article_count:,}")
    print(f"   📊 Total characters: {total_chars:,}")
    print(f"   💾 Saved to: {output_dir}")

    return article_count, total_chars


def main():
    parser = argparse.ArgumentParser(description="Extract Wikipedia articles from XML dump")
    parser.add_argument("xml_path", type=Path, help="Path to Wikipedia .xml.bz2 file")
    parser.add_argument("output_dir", type=Path, help="Output directory for extracted articles")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Maximum number of articles to extract")
    parser.add_argument("--min-length", type=int, default=500,
                        help="Minimum article length in characters (default: 500)")

    args = parser.parse_args()

    if not args.xml_path.exists():
        print(f"❌ Error: File not found: {args.xml_path}")
        return

    extract_articles(
        args.xml_path,
        args.output_dir,
        max_articles=args.max_articles,
        min_length=args.min_length
    )


if __name__ == "__main__":
    main()
