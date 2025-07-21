#!/usr/bin/env python3
"""
Streaming Data Loader - Load training data from APIs without local storage
"""

import requests
import json
import time
from typing import Iterator, Optional, Dict, Any, List
from datasets import load_dataset
import wikipedia
from itertools import islice
import warnings
from requests.exceptions import RequestException
from urllib3.exceptions import HTTPError

# Suppress warnings from BeautifulSoup and Wikipedia
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

class StreamingDataLoader:
    """Load large datasets from APIs without storing locally"""
    
    def __init__(self, cache_size: int = 1000, 
                 wikipedia_delay: float = 0.1,
                 gutenberg_delay: float = 0.5,
                 max_retries: int = 3,
                 base_backoff_delay: float = 1.0):
        self.cache_size = cache_size
        self._cache = []
        
        # Rate limiting configuration
        self.wikipedia_delay = wikipedia_delay
        self.gutenberg_delay = gutenberg_delay
        self.max_retries = max_retries
        self.base_backoff_delay = base_backoff_delay
        
    def stream_openwebtext(self, num_samples: Optional[int] = None) -> Iterator[str]:
        """Stream OpenWebText dataset from Hugging Face"""
        try:
            # Try HuggingFace OpenWebText dataset
            dataset = load_dataset("openwebtext", streaming=True, split="train")
            count = 0
            
            for item in dataset:
                if num_samples and count >= num_samples:
                    break
                yield item['text']
                count += 1
                
        except Exception as e:
            print(f"Error streaming OpenWebText: {e}")
            # Fallback to alternative dataset if primary fails
            try:
                print("Trying alternative OpenWebText dataset...")
                dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
                count = 0
                for item in dataset:
                    if num_samples and count >= num_samples:
                        break
                    yield item['text']
                    count += 1
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                # Third fallback: use a simple text generator
                print("Using simple text fallback...")
                for i in range(num_samples or 100):
                    yield f"This is sample text {i+1} for training purposes. " * 10
    
    def stream_pile(self, subset: str = "all", num_samples: Optional[int] = None) -> Iterator[str]:
        """Stream The Pile dataset from Hugging Face"""
        try:
            dataset = load_dataset("EleutherAI/pile", streaming=True, split="train")
            count = 0
            
            for item in dataset:
                if num_samples and count >= num_samples:
                    break
                yield item['text']
                count += 1
                
        except Exception as e:
            print(f"Error streaming The Pile: {e}")
            return
    
    def stream_c4(self, num_samples: Optional[int] = None) -> Iterator[str]:
        """Stream C4 dataset from Hugging Face"""
        try:
            dataset = load_dataset("c4", "en", streaming=True, split="train")
            count = 0
            
            for item in dataset:
                if num_samples and count >= num_samples:
                    break
                yield item['text']
                count += 1
                
        except Exception as e:
            print(f"Error streaming C4: {e}")
            return
    
    def stream_wikipedia(self, language: str = "en", num_articles: Optional[int] = None) -> Iterator[str]:
        """Stream Wikipedia articles via API"""
        try:
            wikipedia.set_lang(language)
            
            # Get random article titles
            count = 0
            while True:
                if num_articles and count >= num_articles:
                    break
                    
                try:
                    # Get random articles
                    titles = wikipedia.random(10)
                    
                    for title in titles:
                        if num_articles and count >= num_articles:
                            break
                            
                        try:
                            page = wikipedia.page(title)
                            yield page.content
                            count += 1
                            
                        except (wikipedia.exceptions.DisambiguationError, 
                               wikipedia.exceptions.PageError):
                            continue
                            
                        # Rate limiting with configurable delay
                        time.sleep(self.wikipedia_delay)
                        
                except Exception as e:
                    print(f"Error fetching Wikipedia articles: {e}")
                    time.sleep(self.base_backoff_delay)
                    continue
                    
        except Exception as e:
            print(f"Error setting up Wikipedia streaming: {e}")
            return
    
    def stream_gutenberg(self, num_books: Optional[int] = None) -> Iterator[str]:
        """Stream books from Project Gutenberg via API"""
        try:
            # Project Gutenberg metadata API
            base_url = "https://gutendex.com/books/"
            count = 0
            page = 1
            
            while True:
                if num_books and count >= num_books:
                    break
                    
                try:
                    # Get book metadata with rate limit handling
                    response = self._make_request_with_retry(f"{base_url}?page={page}")
                    if response is None:
                        break
                    data = response.json()
                    
                    if not data.get('results'):
                        break
                    
                    for book in data['results']:
                        if num_books and count >= num_books:
                            break
                            
                        # Find plain text format
                        text_url = None
                        for format_type, url in book.get('formats', {}).items():
                            if 'text/plain' in format_type:
                                text_url = url
                                break
                        
                        if text_url:
                            try:
                                text_response = self._make_request_with_retry(text_url)
                                if text_response is None:
                                    continue
                                
                                # Clean and yield text
                                text = text_response.text
                                if len(text) > 1000:  # Filter very short texts
                                    yield text
                                    count += 1
                                    
                            except Exception as e:
                                print(f"Error downloading book {book.get('title', 'Unknown')}: {e}")
                                continue
                        
                        # Rate limiting with configurable delay
                        time.sleep(self.gutenberg_delay)
                    
                    page += 1
                    
                except Exception as e:
                    print(f"Error fetching Gutenberg page {page}: {e}")
                    break
                    
        except Exception as e:
            print(f"Error setting up Gutenberg streaming: {e}")
            return
    
    def _make_request_with_retry(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """Make HTTP request with exponential backoff and rate limit handling"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = self._get_retry_after_delay(response)
                    print(f"🚦 Rate limited. Waiting {retry_after}s before retry...")
                    time.sleep(retry_after)
                    continue
                    
                # Check for other HTTP errors
                response.raise_for_status()
                return response
                
            except (RequestException, HTTPError) as e:
                delay = self.base_backoff_delay * (2 ** attempt)
                print(f"⚠️ Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    print(f"🔄 Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"❌ Max retries exceeded for {url}")
                    return None
                    
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return None
        
        return None
    
    def _get_retry_after_delay(self, response: requests.Response) -> float:
        """Extract retry-after delay from response headers"""
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        
        # Default delay if no Retry-After header
        return self.base_backoff_delay * 2
    
    def stream_mixed_sources(self, 
                           sources: Dict[str, int], 
                           total_samples: Optional[int] = None) -> Iterator[str]:
        """
        Stream from multiple sources with specified proportions
        
        Args:
            sources: Dict mapping source names to number of samples
                    e.g., {"openwebtext": 1000, "wikipedia": 500, "pile": 2000}
            total_samples: Optional limit on total samples
        """
        
        iterators = {}
        
        # Initialize iterators for each source
        for source, count in sources.items():
            if source == "openwebtext":
                iterators[source] = self.stream_openwebtext(count)
            elif source == "pile":
                iterators[source] = self.stream_pile(num_samples=count)
            elif source == "c4":
                iterators[source] = self.stream_c4(count)
            elif source == "wikipedia":
                iterators[source] = self.stream_wikipedia(num_articles=count)
            elif source == "gutenberg":
                iterators[source] = self.stream_gutenberg(count)
        
        # Round-robin through sources
        total_yielded = 0
        active_sources = list(iterators.keys())
        
        while active_sources and (not total_samples or total_yielded < total_samples):
            for source in active_sources[:]:  # Copy list to avoid modification during iteration
                try:
                    text = next(iterators[source])
                    yield text
                    total_yielded += 1
                    
                    if total_samples and total_yielded >= total_samples:
                        break
                        
                except StopIteration:
                    active_sources.remove(source)
                except Exception as e:
                    print(f"⚠️ Error from {source}: {e}")
                    active_sources.remove(source)

def create_streaming_text_generator(sources: Dict[str, int], 
                                  chunk_size: int = 10000) -> Iterator[str]:
    """
    Create a streaming text generator that yields large text chunks
    
    Args:
        sources: Dict mapping source names to sample counts
        chunk_size: Target size of each text chunk in characters
    """
    
    loader = StreamingDataLoader()
    text_stream = loader.stream_mixed_sources(sources)
    
    current_chunk = ""
    
    for text in text_stream:
        current_chunk += text + "\n\n"
        
        if len(current_chunk) >= chunk_size:
            yield current_chunk
            current_chunk = ""
    
    # Yield remaining text
    if current_chunk.strip():
        yield current_chunk

# Example usage and testing
if __name__ == "__main__":
    # Test streaming from different sources
    loader = StreamingDataLoader()
    
    print("Testing Wikipedia streaming...")
    wiki_stream = loader.stream_wikipedia(num_articles=2)
    for i, text in enumerate(wiki_stream):
        print(f"Article {i+1}: {len(text)} chars")
        print(text[:200] + "...\n")
    
    print("\nTesting mixed sources...")
    mixed_sources = {
        "wikipedia": 2,
        "openwebtext": 2
    }
    
    mixed_stream = loader.stream_mixed_sources(mixed_sources)
    for i, text in enumerate(mixed_stream):
        print(f"Sample {i+1}: {len(text)} chars")
        print(text[:100] + "...\n")