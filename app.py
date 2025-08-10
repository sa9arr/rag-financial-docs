import streamlit as st
from rag_services.kg_rag_service import answer_query
from rag_services.vector_rag_service import answer_vector_rag_query

st.set_page_config(page_title="Hybrid RAG App", layout="centered")

st.title("🔍 RAG with APPL Financial Docs")
st.write("Ask a question and choose a retrieval method (Vector or Knowledge Graph).")
st.write("Vector RAG uses ChromaDB and FinLang/finance-embeddings-investopedia, while Knowledge Graph RAG uses LightRAG with OpenAI embeddings.")

# Radio button to choose the RAG method
rag_method = st.radio(
    "Choose a RAG method:",
    ("KG RAG", "Vector RAG")
)

# Input field
query = st.text_input("Enter your question:")

# Submit button
if st.button("Search"):
    if query.strip():
        with st.spinner(f"Retrieving answer using {rag_method}..."):
            try:
                if rag_method == "KG RAG":
                    answer = answer_query(query)
                elif rag_method == "Vector RAG":
                    answer = answer_vector_rag_query(query)
                
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")