import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


load_dotenv()

def load_and_chunk_markdown(file_path: str) -> list[Document]:
    """Loads and splits a markdown file into chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    print(f"Loading markdown file: {file_path}")
    loader = TextLoader(file_path)
    documents = loader.load()
    

    # splitting according to the structure of the markdown file

     # Using RecursiveCharacterTextSplitter to create semantically relevant chunks
    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    
    print("Splitting markdown into chunks...")
    chunks = markdown_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    return chunks

# this embedding model if fine tuned on bge-large with the financial datasets
def create_rag_pipeline(chunks: list[Document], embedding_model_name: str = 'FinLang/finance-embeddings-investopedia'): 
    """Creates and indexes the vector database with the given chunks."""
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    print("Creating and indexing vector database with ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    db.persist()
    print("Indexing complete. The vector database is saved to the './chroma_db' directory.")
    return db

if __name__ == "__main__":
  
    markdown_file_path = "/home/ozymas/Pictures/rag-financial-docs/aapl_financial_data/2022 Q3 AAPL.md"
    
    try:
        # Step 1: Load and chunk the markdown file
        chunks = load_and_chunk_markdown(markdown_file_path)
        
        # Step 2 & 3: Create and index the vector database
        db = create_rag_pipeline(chunks)
    
    except FileNotFoundError as e:
        print(e)
        print("Please ensure your markdown file is in the 'data' folder.")