import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def answer_vector_rag_query(query: str):
    """
    Performs a query on the vector RAG pipeline to get a final answer.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    

    embedding_model = HuggingFaceEmbeddings(model_name='FinLang/finance-embeddings-investopedia')
    
    # Load the existing ChromaDB from disk
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=openai_api_key)

    prompt = ChatPromptTemplate.from_template("""
        You are an expert financial analyst. Your task is to provide accurate and factual answers to user questions based *only* on the financial report data provided in the context.

            Follow these strict rules:

            1.  **Do not hallucinate.** If the exact information to answer the question is not explicitly present in the provided context, state clearly that you cannot find the information.
            2.  **Be concise and direct.** Provide the answer in a straightforward manner, without adding extraneous details or conversational filler.
            3.  **Quote numbers and facts precisely.** When asked for a specific financial figure, revenue, or date, state it exactly as it appears in the context.
            4.  **Adhere to the persona.** Maintain the tone of a professional financial analyst. Do not use informal language or personal opinions.

            **Context:**
            {context}

            **Question:**
            {input}

            **Answer:**
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    try:
        response = retrieval_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"An error occurred: {e}"