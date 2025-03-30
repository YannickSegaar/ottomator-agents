"""
Factory for creating agents from configuration.
This centralizes agent creation logic to make adding new agents easier.
"""

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
from typing import List, Dict, Any, Set, Optional, Union

# Load agent configurations
from agent_config import AGENTS, COMBINED_AGENTS

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AgentDeps:
    """Dependencies required by all agents."""
    supabase: Client
    openai_client: AsyncOpenAI
    source_tags: List[str]  # List of source tags to search in

# Cache for created agents
_agent_cache = {}

def create_agent(agent_key: str) -> Agent:
    """
    Creates an agent based on the configuration key.
    Uses a cache to avoid recreating agents.
    
    Args:
        agent_key: Key of the agent in the configuration
        
    Returns:
        The created agent
    """
    # Return from cache if already created
    if agent_key in _agent_cache:
        return _agent_cache[agent_key]
        
    # Determine if this is a single or combined agent
    if agent_key in AGENTS:
        config = AGENTS[agent_key]
        source_tags = [config["source_tag"]]
    elif agent_key in COMBINED_AGENTS:
        config = COMBINED_AGENTS[agent_key]
        source_tags = config["source_tags"]
    else:
        raise ValueError(f"Unknown agent key: {agent_key}")
    
    # Create the agent with tools
    agent = Agent(
        model,
        system_prompt=config["system_prompt"],
        deps_type=AgentDeps,
        retries=2
    )
    
    # Add tools to the agent
    add_tools_to_agent(agent, source_tags)
    
    # Cache the agent
    _agent_cache[agent_key] = agent
    
    return agent

def add_tools_to_agent(agent: Agent, source_tags: List[str]):
    """
    Adds standard tools to an agent.
    
    Args:
        agent: The agent to add tools to
        source_tags: List of source tags this agent should search in
    """
    
    @agent.tool
    async def retrieve_relevant_documentation(ctx: RunContext[AgentDeps], user_query: str, specific_source: str = None) -> str:
        """
        Retrieve relevant documentation chunks based on the query with RAG.
        
        Args:
            ctx: The context including the Supabase client and OpenAI client
            user_query: The user's question or query
            specific_source: Optional specific source to search in
            
        Returns:
            A formatted string containing the most relevant documentation chunks
        """
        try:
            # Get the embedding for the query
            query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
            
            # Determine which sources to search in
            search_sources = [specific_source] if specific_source else ctx.deps.source_tags
            
            all_results = []
            formatted_sections = []
            debug_info = []
            
            # For each source, perform a search
            for source in search_sources:
                debug_info.append(f"Searching source: {source}")
                
                # Query Supabase for relevant documents - get more results (8) to ensure coverage
                result = ctx.deps.supabase.rpc(
                    'match_site_pages',
                    {
                        'query_embedding': query_embedding,
                        'match_count': 8,  # Increased from 3 to 8 for better coverage
                        'filter': {'source': source}
                    }
                ).execute()
                
                debug_info.append(f"Found {len(result.data) if result.data else 0} vector results for {source}")
                
                if result.data:
                    # Group results by source
                    source_results = []
                    for doc in result.data:
                        chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                        source_results.append(chunk_text)
                        all_results.append(doc)
                    
                    # Add a formatted section for this source
                    if source_results:
                        section_title = f"## {source.replace('_docs', '').title()} Documentation"
                        formatted_content = section_title + "\n\n" + "\n\n---\n\n".join(source_results)
                        formatted_sections.append(formatted_content)
            
            # If vector search failed, try direct text search
            if not all_results:
                debug_info.append("Vector search found no results. Trying direct text search.")
                
                # Break query into keywords for better text search
                clean_query = user_query.replace("?", "").replace("!", "").lower()
                keywords = [word for word in clean_query.split() if len(word) > 3]
                
                # Create search terms with exact phrases and individual keywords
                search_terms = [f"'{user_query}'"]  # Exact phrase match
                search_terms.extend(keywords)  # Add individual keywords
                
                # Join with | for OR logic in text search
                text_search_query = " | ".join(search_terms)
                debug_info.append(f"Text search query: {text_search_query}")
                
                for source in search_sources:
                    # Use direct text search approach
                    try:
                        # Direct content search
                        term_search = ctx.deps.supabase.from_('site_pages') \
                            .select('title, content, url') \
                            .eq('metadata->>source', source) \
                            .textearch('content', text_search_query) \
                            .limit(5) \
                            .execute()
                            
                        debug_info.append(f"Found {len(term_search.data) if term_search.data else 0} text search results for {source}")
                        
                        if term_search.data:
                            section_title = f"## {source.replace('_docs', '').title()} Documentation (Text Search)"
                            source_results = []
                            for doc in term_search.data:
                                chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                                source_results.append(chunk_text)
                            
                            formatted_content = section_title + "\n\n" + "\n\n---\n\n".join(source_results)
                            formatted_sections.append(formatted_content)
                    except Exception as text_error:
                        debug_info.append(f"Text search error: {str(text_error)}")
                        
                        # Fallback to basic keyword search if textearch doesn't work
                        try:
                            # Try a simpler approach - look for content containing any of the keywords
                            for keyword in keywords:
                                if len(keyword) < 4:
                                    continue  # Skip very short words
                                    
                                keyword_search = ctx.deps.supabase.from_('site_pages') \
                                    .select('title, content, url') \
                                    .eq('metadata->>source', source) \
                                    .ilike('content', f'%{keyword}%') \
                                    .limit(3) \
                                    .execute()
                                    
                                if keyword_search.data:
                                    debug_info.append(f"Found {len(keyword_search.data)} results for keyword '{keyword}'")
                                    section_title = f"## {source.replace('_docs', '').title()} Documentation (Keyword: {keyword})"
                                    source_results = []
                                    for doc in keyword_search.data:
                                        chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                                        source_results.append(chunk_text)
                                    
                                    formatted_content = section_title + "\n\n" + "\n\n---\n\n".join(source_results)
                                    formatted_sections.append(formatted_content)
                                    break  # Exit after finding results for one keyword
                        except Exception as keyword_error:
                            debug_info.append(f"Keyword search error: {str(keyword_error)}")
            
            # Special search for "feature matrix" or "support matrix" if that's in the query
            if "feature matrix" in user_query.lower() or "support matrix" in user_query.lower() or "compatibility" in user_query.lower():
                debug_info.append("Detected feature/support matrix request - doing targeted search")
                for source in search_sources:
                    matrix_search = ctx.deps.supabase.from_('site_pages') \
                        .select('title, content, url') \
                        .eq('metadata->>source', source) \
                        .or_(f"content.ilike.%feature matrix%, content.ilike.%support matrix%, content.ilike.%compatibility table%") \
                        .limit(3) \
                        .execute()
                        
                    if matrix_search.data:
                        debug_info.append(f"Found {len(matrix_search.data)} matrix results")
                        section_title = f"## {source.replace('_docs', '').title()} Matrix Information"
                        source_results = []
                        for doc in matrix_search.data:
                            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                            source_results.append(chunk_text)
                        
                        formatted_content = section_title + "\n\n" + "\n\n---\n\n".join(source_results)
                        formatted_sections.append(formatted_content)
            
            if not formatted_sections:
                # Include debug info in the response if no results were found
                debug_str = "\n".join(debug_info)
                return f"No relevant documentation found for your query.\n\nDiagnostic information:\n{debug_str}"
                
            # Join all sections with a major separator
            result = "\n\n==========\n\n".join(formatted_sections)
            
            # Include some debug info at the end for troubleshooting
            debug_str = "\n".join(debug_info)
            return result
            
        except Exception as e:
            print(f"Error retrieving documentation: {e}")
            return f"Error retrieving documentation: {str(e)}"
    
    @agent.tool
    async def direct_keyword_search(ctx: RunContext[AgentDeps], keyword: str, specific_source: str = None) -> str:
        """
        Directly search for a specific keyword or phrase in the documentation.
        Useful when semantic search isn't finding the right content.
        
        Args:
            ctx: The context
            keyword: The exact keyword or phrase to search for
            specific_source: Optional source to limit search to
            
        Returns:
            Documentation chunks containing the keyword
        """
        try:
            # Determine which sources to search in
            search_sources = [specific_source] if specific_source else ctx.deps.source_tags
            
            all_results = []
            formatted_sections = []
            
            for source in search_sources:
                # Simple keyword search
                keyword_search = ctx.deps.supabase.from_('site_pages') \
                    .select('title, content, url') \
                    .eq('metadata->>source', source) \
                    .ilike('content', f'%{keyword}%') \
                    .limit(5) \
                    .execute()
                    
                if keyword_search.data:
                    section_title = f"## {source.replace('_docs', '').title()} Documentation (Keyword: {keyword})"
                    source_results = []
                    for doc in keyword_search.data:
                        chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                        source_results.append(chunk_text)
                    
                    formatted_content = section_title + "\n\n" + "\n\n---\n\n".join(source_results)
                    formatted_sections.append(formatted_content)
            
            if not formatted_sections:
                return f"No documentation found containing the keyword: '{keyword}'"
                
            # Join all sections with a separator
            return "\n\n==========\n\n".join(formatted_sections)
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return f"Error performing keyword search: {str(e)}"
    
    @agent.tool
    async def list_documentation_pages(ctx: RunContext[AgentDeps], specific_source: str = None) -> Dict[str, List[str]]:
        """
        Retrieve a list of all available documentation pages.
        
        Args:
            ctx: The context including the Supabase client
            specific_source: Optional specific source to list pages for
            
        Returns:
            Dict with keys for each source, each containing a list of URLs
        """
        try:
            result = {}
            
            # Determine which sources to list pages for
            sources_to_list = [specific_source] if specific_source else ctx.deps.source_tags
            
            for source in sources_to_list:
                source_result = ctx.deps.supabase.from_('site_pages') \
                    .select('url') \
                    .eq('metadata->>source', source) \
                    .execute()
                    
                if source_result.data:
                    result[source] = sorted(set(doc['url'] for doc in source_result.data))
                else:
                    result[source] = []
            
            return result
            
        except Exception as e:
            print(f"Error retrieving documentation pages: {e}")
            return {source: [] for source in sources_to_list}
    
    @agent.tool
    async def get_page_content(ctx: RunContext[AgentDeps], url: str, specific_source: str = None) -> str:
        """
        Retrieve the full content of a specific documentation page by combining all its chunks.
        
        Args:
            ctx: The context including the Supabase client
            url: The URL of the page to retrieve
            specific_source: Optional specific source to retrieve from
            
        Returns:
            str: The complete page content with all chunks combined in order
        """
        try:
            # If source is specified, only search in that source
            query = ctx.deps.supabase.from_('site_pages') \
                .select('title, content, chunk_number, metadata') \
                .eq('url', url)
                
            if specific_source:
                query = query.eq('metadata->>source', specific_source)
                
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

def get_available_agents() -> Dict[str, Dict]:
    """
    Get all available agents for display in the UI.
    
    Returns:
        Dict of agent configurations
    """
    # Combine single and combined agents
    all_agents = {}
    
    # Add individual agents first
    for key, config in AGENTS.items():
        all_agents[key] = config
    
    # Add combined agents
    for key, config in COMBINED_AGENTS.items():
        all_agents[key] = config
    
    return all_agents