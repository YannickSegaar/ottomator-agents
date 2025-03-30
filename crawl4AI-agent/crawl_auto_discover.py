import os
import sys
import json
import asyncio
import re
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

async def auto_discover_crawl(start_url: str, source_tag: str, url_filter: str = None, max_pages: int = 100):
    """
    Crawl a website starting from a URL and automatically discover linked pages.
    
    Args:
        start_url: The starting URL for crawling
        source_tag: Metadata tag to identify the documentation source
        url_filter: Regex pattern to filter URLs (e.g., '/docs/')
        max_pages: Maximum number of pages to crawl
    """
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Queue of URLs to crawl
        to_crawl = [start_url]
        # Set of already crawled URLs
        crawled = set()
        # Set of all discovered URLs
        discovered = set(to_crawl)
        
        # Process URLs until queue is empty or max_pages is reached
        while to_crawl and len(crawled) < max_pages:
            # Get the next URL to crawl
            current_url = to_crawl.pop(0)
            
            # Skip if already crawled
            if current_url in crawled:
                continue
                
            print(f"Crawling {len(crawled)+1}/{max_pages}: {current_url}")
            
            # Crawl the current URL
            result = await crawler.arun(
                url=current_url,
                config=crawl_config,
                session_id="session1"
            )
            
            if result.success:
                print(f"Successfully crawled: {current_url}")
                
                # Process and store document
                await process_and_store_document(current_url, result.markdown_v2.raw_markdown, source_tag)
                
                # Extract links from HTML
                new_urls = extract_links(result.html, current_url, url_filter)
                
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
            
        print(f"Crawling complete. Processed {len(crawled)} pages.")
    finally:
        await crawler.close()

async def main():
    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python crawl_auto_discover.py <start_url> <source_tag> [url_filter] [max_pages]")
        print("Example: python crawl_auto_discover.py https://myfitnesspalapi.com/docs/ myfitnesspal_docs '/docs/' 50")
        return
        
    start_url = sys.argv[1]
    source_tag = sys.argv[2]
    url_filter = sys.argv[3] if len(sys.argv) > 3 else None
    max_pages = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    print(f"Starting auto-discovery crawl from: {start_url}")
    print(f"Source tag: {source_tag}")
    print(f"URL filter: {url_filter or 'None'}")
    print(f"Max pages: {max_pages}")
    
    await auto_discover_crawl(start_url, source_tag, url_filter, max_pages)

if __name__ == "__main__":
    asyncio.run(main())