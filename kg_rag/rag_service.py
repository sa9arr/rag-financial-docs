import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# Keep the RAG object in memory after initialization
_rag_instance = None

async def _initialize_rag():
    """Initialize the LightRAG instance asynchronously."""
    rag = LightRAG(
        working_dir="../aapl_financial_data", 
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        enable_llm_cache=False
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def get_rag():
    """Returns a cached LightRAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = asyncio.run(_initialize_rag())
    return _rag_instance

def answer_query(user_query: str):
    """Queries the knowledge graph and returns the response."""
    rag = get_rag()
    query_param = QueryParam(mode="mix", only_need_context=False)
    response = rag.query(user_query, param=query_param)
    return response
