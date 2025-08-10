import streamlit as st
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status


_rag_instance = None

async def _initialize_rag():
    """Initialize the LightRAG instance asynchronously."""
    rag = LightRAG(
        working_dir="./aapl_financial_data",  
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        enable_llm_cache=False
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def get_rag():
    """Returns a cached LightRAG instance, initializes if not already."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = asyncio.run(_initialize_rag())
    return _rag_instance

def answer_query(user_query: str):
    """Queries the knowledge graph and returns the response."""
    rag = get_rag()
    query_param = QueryParam(mode="mix", only_need_context=False)
    return rag.query(user_query, param=query_param)




st.set_page_config(page_title="KG Retrieval App", layout="centered")

st.title("🔍 Knowledge Graph Retrieval App")
st.write("Ask a question and get an AI-generated answer from your indexed knowledge base.")

# Input field
query = st.text_input("Enter your question:")

# Search button
if st.button("Search"):
    if query.strip():
        with st.spinner("Retrieving answer..."):
            try:
                answer = answer_query(query)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
