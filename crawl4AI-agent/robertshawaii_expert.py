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
class RobertsHawaiiDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert on Roberts Hawaii, the premier tour and transportation company in Hawaii.

You have detailed knowledge about:
- All tours and activities offered by Roberts Hawaii across all Hawaiian islands
- Transportation services including airport shuttles and group transportation
- Island-specific tours for Oahu, Maui, Big Island, and Kauai
- Pricing, booking policies, and logistics
- Popular attractions and destinations included in their tours
- Special packages and offerings
- Practical information such as pickup locations, duration, and what to bring

Your job is to help customers learn about Roberts Hawaii offerings, recommend appropriate tours based on their interests, answer questions about logistics, and provide expert guidance for Hawaiian vacation activities.

Always provide accurate, helpful information based on the official Roberts Hawaii website content. Your answers should match the tone of a friendly, knowledgeable Hawaiian tour representative.

When helping with tour selection, consider factors like the island they're visiting, their interests (nature, culture, adventure), group size, and any specific needs.

Don't ask the user before taking an action like searching for information - just do it. Always look at the documentation with the provided tools before answering questions.

Always be honest if you don't find the specific information in the available documentation.
"""

robertshawaii_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=RobertsHawaiiDeps,
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

@robertshawaii_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[RobertsHawaiiDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 6,  # Get top 6 results for better coverage
                'match_threshold': 0.5,  # Lower threshold for more inclusive matching
                'filter': {'source': 'robertshawaii_docs'}
            }
        ).execute()
        
        if not result.data:
            # Fallback to text search if vector search fails
            clean_query = user_query.replace("?", "").replace("!", "").lower()
            keywords = [word for word in clean_query.split() if len(word) > 3]
            
            all_content = []
            for keyword in keywords:
                if len(keyword) < 4:
                    continue  # Skip very short words
                    
                keyword_search = ctx.deps.supabase.from_('site_pages') \
                    .select('title, content, url') \
                    .eq('metadata->>source', 'robertshawaii_docs') \
                    .ilike('content', f'%{keyword}%') \
                    .limit(3) \
                    .execute()
                    
                if keyword_search.data:
                    all_content.extend(keyword_search.data)
                    
            if not all_content:
                return "I couldn't find specific information about that in the Roberts Hawaii documentation. Can I help you with something else about their tours or services?"
                
            # Format the results
            formatted_chunks = []
            for doc in all_content[:5]:  # Limit to top 5 results
                chunk_text = f"""
# {doc['title']}

{doc['content']}

URL: {doc['url']}
"""
                formatted_chunks.append(chunk_text)
                
            return "\n\n---\n\n".join(formatted_chunks)
            
        # Process vector search results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}

URL: {doc['url']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@robertshawaii_expert.tool
async def search_tours_by_island(ctx: RunContext[RobertsHawaiiDeps], island: str) -> str:
    """
    Search for Roberts Hawaii tours on a specific Hawaiian island.
    
    Args:
        ctx: The context including the Supabase client
        island: The Hawaiian island name (Oahu, Maui, Kauai, Big Island)
        
    Returns:
        A formatted string containing tour options for the specified island
    """
    try:
        # Normalize island name for search
        island_lower = island.lower()
        
        # Handle common variations
        if island_lower == "big island" or island_lower == "hawaii island":
            search_terms = ["big island", "hawaii island", "island of hawaii"]
        else:
            search_terms = [island_lower]
            
        all_results = []
        
        # Search for content mentioning the island
        for term in search_terms:
            term_search = ctx.deps.supabase.from_('site_pages') \
                .select('title, content, url') \
                .eq('metadata->>source', 'robertshawaii_docs') \
                .ilike('content', f'%{term}%') \
                .limit(10) \
                .execute()
                
            if term_search.data:
                all_results.extend(term_search.data)
                
        if not all_results:
            return f"I couldn't find specific tour information for {island}. Would you like information about tours on a different Hawaiian island?"
            
        # Format the results
        formatted_chunks = []
        for doc in all_results[:8]:  # Limit to top 8 results
            chunk_text = f"""
# {doc['title']}

{doc['content']}

URL: {doc['url']}
"""
            formatted_chunks.append(chunk_text)
            
        return f"## Roberts Hawaii Tours on {island}\n\n" + "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error searching tours by island: {e}")
        return f"Error searching tours: {str(e)}"

@robertshawaii_expert.tool
async def search_by_activity_type(ctx: RunContext[RobertsHawaiiDeps], activity_type: str) -> str:
    """
    Search for Roberts Hawaii tours by activity type.
    
    Args:
        ctx: The context including the Supabase client
        activity_type: Type of activity (e.g., snorkeling, hiking, luau, sightseeing)
        
    Returns:
        A formatted string containing tours matching the activity type
    """
    try:
        # Direct search for the activity type
        activity_search = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, url') \
            .eq('metadata->>source', 'robertshawaii_docs') \
            .ilike('content', f'%{activity_type}%') \
            .limit(8) \
            .execute()
            
        if not activity_search.data:
            return f"I couldn't find specific information about {activity_type} activities with Roberts Hawaii. Would you like to know about other types of tours they offer?"
            
        # Format the results
        formatted_chunks = []
        for doc in activity_search.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}

URL: {doc['url']}
"""
            formatted_chunks.append(chunk_text)
            
        return f"## Roberts Hawaii {activity_type.title()} Activities\n\n" + "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error searching by activity type: {e}")
        return f"Error searching activities: {str(e)}"

@robertshawaii_expert.tool
async def list_documentation_pages(ctx: RunContext[RobertsHawaiiDeps]) -> List[str]:
    """
    Retrieve a list of all available Roberts Hawaii documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'robertshawaii_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@robertshawaii_expert.tool
async def get_page_content(ctx: RunContext[RobertsHawaiiDeps], url: str) -> str:
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
            .eq('metadata->>source', 'robertshawaii_docs') \
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