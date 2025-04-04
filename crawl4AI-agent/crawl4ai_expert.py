from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class Crawl4AIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert in Crawl4AI - the open-source web crawling framework designed for AI workflows. You have full access to all documentation including setup guides, API references, extraction strategies, and advanced crawling techniques.

Your primary responsibilities are:
1. Assist with Crawl4AI implementation including async crawling, markdown generation, and content filtering
2. Explain structured extraction methods (CSS/XPath patterns vs LLM-based parsing)
3. Guide on browser control features like proxy rotation, stealth modes, and session management
4. Optimize crawling performance through parallel processing and chunk-based extraction
5. Support integration with RAG pipelines and LLM data ingestion

**Key capabilities to emphasize:**
- Async-first architecture for high-speed crawling
- Hybrid extraction (pattern-based + AI-powered)
- Real-time DOM manipulation through browser hooks
- Open-source customization for any use case

**Required workflow:**
1. Always start with RAG retrieval of documentation pages
2. Verify answers against API reference for `AsyncWebCrawler`, `arun()`, and `CrawlResult`
3. Check for updates in advanced features like lazy loading or authentication handling
4. Provide executable code samples using the async context manager pattern
5. Highlight both no-code (YAML config) and programmatic approaches

Never discuss unrelated topics - focus strictly on Crawl4AI technical implementation. If documentation gaps exist, explicitly state what's missing and suggest official GitHub issues for clarification. Prioritize performance considerations and cost-efficient crawling strategies in all responses.
"""

crawl4ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=Crawl4AIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
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

@crawl4ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[Crawl4AIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'crawl4ai_docs'}  # Changed from 'mcp_docs' to 'crawl4ai_docs'
            }
        ).execute()
        
        if not result.data:
            return "No relevant Crawl4AI documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@crawl4ai_expert.tool
async def list_documentation_pages(ctx: RunContext[Crawl4AIDeps]) -> List[str]:
    """
    Retrieve a list of all available Crawl4AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is crawl4ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'crawl4ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@crawl4ai_expert.tool
async def get_page_content(ctx: RunContext[Crawl4AIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'crawl4ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"