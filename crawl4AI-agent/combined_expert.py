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
from typing import List, Dict, Any

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class CombinedDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert on both the Model Context Protocol (MCP) and the MyFitnessPal API.

For MCP:
- You understand how to implement the Model Context Protocol
- You know the technical specifications and best practices
- You can explain how to use it to provide models with structured context

For MyFitnessPal API:
- You know all the available endpoints, authentication methods, and data models
- You understand how to use the API to access nutrition and fitness data
- You can help with implementation details and practical use cases

Your goal is to help users integrate these technologies, especially for creating an MCP server that uses MyFitnessPal APIs to create an AI agent for nutrition assistance. You can combine knowledge from both domains to suggest solutions.

Always look at the relevant documentation with the provided tools before answering. When searching, you'll automatically search both documentation sources.

Always let the user know when you didn't find the answer in the documentation - be honest.
"""

combined_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=CombinedDeps,
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

@combined_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[CombinedDeps], user_query: str, source: str = None) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        source: Optional filter for documentation source ('mcp_docs' or 'myfitnesspal_docs'). If None, searches both.
        
    Returns:
        A formatted string containing the top relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Set up the filter based on source parameter
        filter_obj = {}
        if source:
            filter_obj = {'source': source}
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 6,  # Increased to get more results when searching across both sources
                'filter': filter_obj
            }
        ).execute()
        
        if not result.data:
            return f"No relevant documentation found for your query."
        
        # Group results by source
        mcp_chunks = []
        mfp_chunks = []
        
        for doc in result.data:
            source = doc.get('metadata', {}).get('source')
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            if source == 'mcp_docs':
                mcp_chunks.append(chunk_text)
            elif source == 'myfitnesspal_docs':
                mfp_chunks.append(chunk_text)
        
        # Prepare the formatted response
        formatted_response = []
        
        if mcp_chunks:
            formatted_response.append("## MCP Documentation\n" + "\n\n---\n\n".join(mcp_chunks[:3]))
            
        if mfp_chunks:
            formatted_response.append("## MyFitnessPal API Documentation\n" + "\n\n---\n\n".join(mfp_chunks[:3]))
            
        if not formatted_response:
            return "No relevant documentation found."
            
        # Join all sections with a major separator
        return "\n\n==========\n\n".join(formatted_response)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@combined_expert.tool
async def search_mcp_docs(ctx: RunContext[CombinedDeps], user_query: str) -> str:
    """
    Search specifically in the MCP documentation.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        Relevant MCP documentation
    """
    return await retrieve_relevant_documentation(ctx, user_query, 'mcp_docs')

@combined_expert.tool
async def search_myfitnesspal_docs(ctx: RunContext[CombinedDeps], user_query: str) -> str:
    """
    Search specifically in the MyFitnessPal API documentation.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        Relevant MyFitnessPal API documentation
    """
    return await retrieve_relevant_documentation(ctx, user_query, 'myfitnesspal_docs')

@combined_expert.tool
async def list_documentation_pages(ctx: RunContext[CombinedDeps], source: str = None) -> Dict[str, List[str]]:
    """
    Retrieve a list of all available documentation pages.
    
    Args:
        ctx: The context including the Supabase client
        source: Optional filter for documentation source ('mcp_docs' or 'myfitnesspal_docs'). If None, lists both.
        
    Returns:
        Dict with keys 'mcp_docs' and 'myfitnesspal_docs', each containing a list of URLs
    """
    try:
        result = {}
        
        # Get MCP docs if requested or if no specific source
        if source is None or source == 'mcp_docs':
            mcp_result = ctx.deps.supabase.from_('site_pages') \
                .select('url') \
                .eq('metadata->>source', 'mcp_docs') \
                .execute()
                
            if mcp_result.data:
                result['mcp_docs'] = sorted(set(doc['url'] for doc in mcp_result.data))
            else:
                result['mcp_docs'] = []
        
        # Get MyFitnessPal docs if requested or if no specific source
        if source is None or source == 'myfitnesspal_docs':
            mfp_result = ctx.deps.supabase.from_('site_pages') \
                .select('url') \
                .eq('metadata->>source', 'myfitnesspal_docs') \
                .execute()
                
            if mfp_result.data:
                result['myfitnesspal_docs'] = sorted(set(doc['url'] for doc in mfp_result.data))
            else:
                result['myfitnesspal_docs'] = []
        
        return result
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return {'mcp_docs': [], 'myfitnesspal_docs': []}

@combined_expert.tool
async def get_page_content(ctx: RunContext[CombinedDeps], url: str, source: str = None) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        source: Optional source identifier ('mcp_docs' or 'myfitnesspal_docs'). If None, tries both.
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # If source is specified, only search in that source
        query = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number, metadata') \
            .eq('url', url)
            
        if source:
            query = query.eq('metadata->>source', source)
            
        result = query.order('chunk_number').execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Get the source from the first result
        first_doc = result.data[0]
        actual_source = first_doc.get('metadata', {}).get('source', 'unknown')
        
        # Format the page with its title and all chunks
        page_title = first_doc['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title} (from {actual_source})\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"