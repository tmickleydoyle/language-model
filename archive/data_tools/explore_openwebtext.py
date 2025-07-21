#!/usr/bin/env python3
"""
Explore OpenWebText data samples.

This script shows you what the actual OpenWebText data looks like.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.streaming_data_loader import StreamingDataLoader

def show_openwebtext_samples(num_samples=5):
    """Show sample OpenWebText data."""
    print("🌐 OpenWebText Data Samples")
    print("=" * 60)
    
    # Create streaming data loader
    loader = StreamingDataLoader()
    
    print("🔄 Fetching OpenWebText samples...")
    
    try:
        # Get samples from OpenWebText
        samples = loader.get_openwebtext_samples(num_samples)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n📄 SAMPLE {i}:")
            print("-" * 40)
            
            # Show first 500 characters
            text = sample[:500]
            print(text)
            
            if len(sample) > 500:
                print(f"\n... (truncated, full length: {len(sample)} chars)")
            
            print(f"\nCharacter count: {len(sample)}")
            print(f"Word count: ~{len(sample.split())}")
            
            # Show some metadata
            if '\n' in sample:
                lines = sample.split('\n')
                print(f"Lines: {len(lines)}")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error fetching OpenWebText: {e}")
        print("\n💡 This might happen if:")
        print("- No internet connection")
        print("- OpenWebText API is down") 
        print("- Rate limiting")
        
        # Show example of what typical OpenWebText looks like
        show_example_data()

def show_example_data():
    """Show example of typical OpenWebText content."""
    print("\n📚 TYPICAL OPENWEBTEXT EXAMPLES:")
    print("=" * 60)
    
    examples = [
        {
            "title": "News Article",
            "text": """Breaking: Scientists Discover New Species in Deep Ocean

Researchers from the Marine Biology Institute have announced the discovery of a previously unknown species of deep-sea creature living at depths of over 3,000 meters. The bioluminescent organism, tentatively named Abyssopelagicus mysterius, exhibits unique characteristics that challenge our understanding of life in extreme environments.

The discovery was made during a three-month expedition using advanced submersible technology. "This finding opens up new questions about biodiversity in the deep ocean," said Dr. Sarah Chen, lead researcher on the project."""
        },
        {
            "title": "Tutorial/How-to",
            "text": """How to Build a Simple Python Web Scraper

Web scraping is a powerful technique for extracting data from websites. In this tutorial, we'll walk through creating a basic web scraper using Python and the BeautifulSoup library.

Prerequisites:
- Basic Python knowledge
- Python 3.6+ installed
- pip package manager

Step 1: Install Required Libraries
First, install the necessary packages:
```bash
pip install requests beautifulsoup4
```

Step 2: Import Libraries
```python
import requests
from bs4 import BeautifulSoup
```"""
        },
        {
            "title": "Blog Post", 
            "text": """The Future of Artificial Intelligence: Opportunities and Challenges

As we stand at the threshold of a new era in artificial intelligence, it's worth reflecting on how far we've come and where we're heading. The rapid advancement of AI technologies over the past decade has been nothing short of remarkable.

From natural language processing to computer vision, AI systems are becoming increasingly sophisticated. However, with these advances come important questions about ethics, employment, and the role of AI in society.

Key areas to watch:
- Autonomous vehicles
- Medical diagnosis
- Climate modeling
- Educational technology"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📖 EXAMPLE {i}: {example['title']}")
        print("-" * 40)
        print(example['text'])
        print(f"\nLength: {len(example['text'])} characters")
        print("-" * 40)

def analyze_data_quality():
    """Analyze what makes OpenWebText good training data."""
    print("\n🔍 WHY OPENWEBTEXT IS GOOD TRAINING DATA:")
    print("=" * 60)
    
    qualities = [
        "📰 **Diverse Content**: News, tutorials, blogs, forums, discussions",
        "🎯 **High Quality**: Human-written, edited content from Reddit links", 
        "📚 **Educational**: How-to guides, explanations, technical content",
        "💬 **Conversational**: Comments, discussions, Q&A formats",
        "🌍 **Broad Topics**: Science, tech, culture, history, etc.",
        "📖 **Well-Structured**: Proper grammar, paragraphs, formatting",
        "🔗 **Contextual**: Articles with references and links",
        "📊 **Large Scale**: Millions of documents for good coverage"
    ]
    
    for quality in qualities:
        print(f"\n{quality}")
    
    print(f"\n💡 **Key Benefits for Language Models:**")
    print("- Learns diverse writing styles and formats")
    print("- Understands different domains and topics") 
    print("- Picks up conversational and formal language")
    print("- Learns to structure information logically")
    print("- Develops reasoning from explanatory content")

def main():
    """Main function."""
    print("🌐 OpenWebText Data Explorer")
    print("=" * 60)
    
    # Show actual samples
    show_openwebtext_samples(3)
    
    # Analyze quality
    analyze_data_quality()
    
    print(f"\n🚀 **For Your Model:**")
    print("With 50,000 OpenWebText samples, your model will learn:")
    print("- Better grammar and sentence structure")
    print("- More diverse vocabulary and topics")
    print("- How to write explanations and tutorials")
    print("- Conversational and formal language patterns")
    print("- Logical information organization")

if __name__ == "__main__":
    main()