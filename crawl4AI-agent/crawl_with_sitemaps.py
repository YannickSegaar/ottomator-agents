import os
import sys
import json
import asyncio
import re
import requests
from xml.etree import ElementTree
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str, source_tag: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": source_tag,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str, source_tag: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url, source_tag) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def get_already_crawled_urls(source_tag: str) -> Set[str]:
    """Get set of URLs that have already been crawled and stored in the database."""
    try:
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', source_tag) \
            .execute()
        
        # Extract unique URLs from the result
        urls = set()
        if result.data:
            for item in result.data:
                urls.add(item['url'])
                
        print(f"Found {len(urls)} already crawled URLs for source: {source_tag}")
        return urls
    except Exception as e:
        print(f"Error retrieving crawled URLs: {e}")
        return set()

def extract_links(html: str, base_url: str, url_filter: str = None) -> List[str]:
    """Extract all links from HTML content and filter by pattern."""
    soup = BeautifulSoup(html, 'html.parser')
    extracted_urls = []
    
    # Parse the base URL to get the domain
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        
        # Skip empty links, javascript actions, or anchor links
        if not href or href.startswith('javascript:') or href.startswith('#'):
            continue
            
        # Convert relative URLs to absolute
        absolute_url = urljoin(base_url, href)
        
        # Parse the URL to get components
        parsed_url = urlparse(absolute_url)
        
        # Skip if different domain or not http(s)
        if parsed_url.netloc != base_domain or not parsed_url.scheme.startswith('http'):
            continue
            
        # Apply URL filter if provided
        if url_filter and not re.search(url_filter, absolute_url):
            continue
            
        extracted_urls.append(absolute_url)
        
    return extracted_urls

def get_urls_from_sitemap(sitemap_url: str, url_filter: str = None) -> List[str]:
    """
    Get URLs from a sitemap or sitemap index.
    Handles both single sitemaps and sitemap indexes that point to multiple sitemaps.
    """
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Check if this is a sitemap index (contains sitemap tags)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemap_tags = root.findall('.//ns:sitemap', namespace)
        
        if sitemap_tags:
            print(f"Found sitemap index with {len(sitemap_tags)} sitemaps")
            all_urls = []
            
            # Process each sitemap in the index
            for sitemap_tag in sitemap_tags:
                sitemap_loc = sitemap_tag.find('./ns:loc', namespace)
                if sitemap_loc is not None and sitemap_loc.text:
                    child_sitemap_url = sitemap_loc.text
                    print(f"Processing sub-sitemap: {child_sitemap_url}")
                    
                    # Recursively get URLs from the child sitemap
                    child_urls = get_urls_from_sitemap(child_sitemap_url, url_filter)
                    all_urls.extend(child_urls)
            
            return all_urls
        else:
            # This is a regular sitemap, extract URLs
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
            
            # Apply URL filter if provided
            if url_filter:
                urls = [url for url in urls if re.search(url_filter, url)]
                
            print(f"Found {len(urls)} URLs in sitemap")
            return urls
    except Exception as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
        return []

async def auto_discover_crawl(start_url: str, source_tag: str, url_filter: str = None, max_pages: int = 100, skip_existing: bool = True, use_sitemap: bool = False):
    """
    Crawl a website starting from a URL and automatically discover linked pages.
    
    Args:
        start_url: The starting URL for crawling or sitemap URL
        source_tag: Metadata tag to identify the documentation source
        url_filter: Regex pattern to filter URLs (e.g., '/docs/')
        max_pages: Maximum number of pages to crawl
        skip_existing: Whether to skip URLs that have already been crawled
        use_sitemap: Whether to use the sitemap at start_url to find pages
    """
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Get already crawled URLs from database if skip_existing is True
    already_crawled = set()
    if skip_existing:
        already_crawled = await get_already_crawled_urls(source_tag)
        print(f"Skipping {len(already_crawled)} already crawled URLs for content processing")

    # Get initial set of URLs to crawl
    to_crawl = []
    
    if use_sitemap:
        print(f"Using sitemap at {start_url} to discover URLs")
        sitemap_urls = get_urls_from_sitemap(start_url, url_filter)
        print(f"Found {len(sitemap_urls)} URLs from sitemap")
        
        # Filter out already crawled URLs if skip_existing is True
        if skip_existing:
            new_urls = [url for url in sitemap_urls if url not in already_crawled]
            print(f"{len(new_urls)} URLs from sitemap are new and will be crawled")
            to_crawl.extend(new_urls)
        else:
            to_crawl.extend(sitemap_urls)
    else:
        # Start with the provided URL
        to_crawl = [start_url]
    
    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Set of all discovered URLs
        discovered = set(to_crawl)
        # Set of already crawled URLs in this run
        crawled = set()
        # Set of URLs we've extracted links from
        extracted_links_from = set()
        
        # Process URLs until queue is empty or max_pages is reached
        pages_crawled = 0
        
        while to_crawl and pages_crawled < max_pages:
            # Get the next URL to crawl
            current_url = to_crawl.pop(0)
            
            # Skip if already crawled in this run
            if current_url in crawled:
                continue
            
            # Check if we should process content or just extract links
            process_content = True
            if skip_existing and current_url in already_crawled:
                if current_url in extracted_links_from:
                    # Skip entirely if we've already extracted links from this URL
                    crawled.add(current_url)
                    continue
                else:
                    # Just extract links without processing content
                    process_content = False
                    print(f"Extracting links only from: {current_url}")
            else:
                print(f"Crawling {pages_crawled+1}/{max_pages}: {current_url}")
            
            # Crawl the current URL
            result = await crawler.arun(
                url=current_url,
                config=crawl_config,
                session_id="session1"
            )
            
            if result.success:
                # Process content if needed
                if process_content:
                    print(f"Successfully crawled: {current_url}")
                    await process_and_store_document(current_url, result.markdown_v2.raw_markdown, source_tag)
                    pages_crawled += 1
                
                # Always extract links
                new_urls = extract_links(result.html, current_url, url_filter)
                
                # Mark that we've extracted links from this URL
                extracted_links_from.add(current_url)
                
                # Add new URLs to the crawl queue if not already discovered
                for url in new_urls:
                    if url not in discovered:
                        to_crawl.append(url)
                        discovered.add(url)
                        print(f"Discovered new URL: {url}")
            else:
                print(f"Failed: {current_url} - Error: {result.error_message}")
            
            # Mark the current URL as crawled
            crawled.add(current_url)
            
            # Wait a bit to be nice to the server
            await asyncio.sleep(1)
            
        print(f"Crawling complete. Processed {pages_crawled} new pages in this run.")
        print(f"Total discovered URLs: {len(discovered)}")
        print(f"URLs remaining in queue: {len(to_crawl)}")
        
        # List some of the remaining URLs if there are any
        if to_crawl:
            print("\nSome remaining URLs in the queue:")
            for url in list(to_crawl)[:10]:  # Show up to 10 URLs
                print(f"- {url}")
            if len(to_crawl) > 10:
                print(f"... and {len(to_crawl) - 10} more")
    finally:
        await crawler.close()

async def main():
    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python crawl_with_sitemaps.py <start_url> <source_tag> [url_filter] [max_pages] [--force] [--sitemap]")
        print("Example (start from page): python crawl_with_sitemaps.py https://example.com/docs/ example_docs '/docs/' 50")
        print("Example (use sitemap): python crawl_with_sitemaps.py https://example.com/sitemap.xml example_docs '/docs/' 50 --sitemap")
        print("Add --force to recrawl all pages, even if they've been crawled before")
        return
        
    start_url = sys.argv[1]
    source_tag = sys.argv[2]
    url_filter = None
    max_pages = 100
    skip_existing = True
    use_sitemap = False
    
    # Parse remaining arguments
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--force":
            skip_existing = False
        elif arg == "--sitemap":
            use_sitemap = True
        elif i == 3 and not arg.startswith("--"):
            url_filter = arg
        elif i == 4 and not arg.startswith("--"):
            try:
                max_pages = int(arg)
            except ValueError:
                print(f"Invalid max_pages value: {arg}. Using default of 100.")
    
    print(f"Starting crawl from: {start_url}")
    print(f"Source tag: {source_tag}")
    print(f"URL filter: {url_filter or 'None'}")
    print(f"Max pages: {max_pages}")
    print(f"Skip existing: {skip_existing}")
    print(f"Use sitemap: {use_sitemap}")
    
    await auto_discover_crawl(start_url, source_tag, url_filter, max_pages, skip_existing, use_sitemap)

if __name__ == "__main__":
    asyncio.run(main())