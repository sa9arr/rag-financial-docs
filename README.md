# Financial Document Analysis with Retrieval-Augmented Generation (RAG)

This repository explores and compares various Retrieval-Augmented Generation (RAG) techniques for extracting insights from financial documents. It provides a hands-on implementation of a hybrid RAG system that leverages both vector-based and knowledge graph-based retrieval methods.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Data Ingestion](#data-ingestion)
  - [Running the Application](#running-the-application)
- [RAG Methods](#rag-methods)
  - [Vector-based RAG](#vector-based-rag)
  - [Knowledge Graph-based RAG](#knowledge-graph-based-rag)

## Project Overview

This project demonstrates how to build and use a RAG system to answer questions about financial reports. It uses Apple's Q3 2022 financial report as a sample document. The primary goal is to showcase how different RAG techniques can be applied to extract and generate insights from complex and unstructured financial data.

## Features

- **Hybrid RAG System:** Implements both vector-based and knowledge graph-based retrieval.
- **Interactive Web Interface:** A simple Streamlit application to interact with the RAG system.
- **Modular Architecture:** The code is organized into separate modules for data ingestion, RAG services, and the user interface.
- **Extensible:** The system can be easily extended to support other documents and RAG techniques.

## System Architecture

The system is composed of the following components:

1.  **Data Ingestion:** The `vector_rag_ingestion.py` script processes the source financial document (in Markdown format), splits it into chunks, and indexes it into a ChromaDB vector store using the `FinLang/finance-embeddings-investopedia` embedding model. The knowledge graph is built using the `LightRAG` library.
2.  **RAG Services:**
    -   `vector_rag_service.py`: Provides a function to answer queries using the vector-based RAG pipeline.
    -   `kg_rag_service.py`: Provides a function to answer queries using the knowledge graph-based RAG pipeline.
3.  **Web Application:** The `app.py` script creates a Streamlit web interface that allows users to ask questions and select the desired RAG method.

## Directory Structure

```
.
├── aapl_financial_data/
│   ├── 2022 Q3 AAPL.md
│   └── ... (other data files)
├── chroma_db/
│   └── ... (ChromaDB vector store)
├── rag_services/
│   ├── kg_rag_service.py
│   ├── vector_rag_ingestion.py
│   └── vector_rag_service.py
├── scripts/
│   ├── kg_based_retrieval.ipynb
│   └── pdf_parsing.ipynb
├── source_data/
│   └── 2022 Q3 AAPL.pdf
├── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/rag-financial-docs.git
    cd rag-financial-docs
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Create a `.env` file in the root directory of the project.
2.  Add your OpenAI API key to the `.env` file:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

## Usage

### Data Ingestion

To ingest the data for the vector-based RAG, run the following command:

```bash
python rag_services/vector_rag_ingestion.py
```

This will process the `aapl_financial_data/2022 Q3 AAPL.md` file and create a ChromaDB vector store in the `chroma_db` directory.

The knowledge graph data is pre-built and stored in the `aapl_financial_data` directory.

### Running the Application

To start the Streamlit web application, run the following command:

```bash
streamlit run app.py
```

This will open a new tab in your browser with the application running. You can then enter a question, choose a RAG method, and get an answer.

## RAG Methods

### Vector-based RAG

This method uses a traditional vector-based approach for retrieval. The documents are chunked and embedded using a sentence transformer model (`FinLang/finance-embeddings-investopedia`) and stored in a ChromaDB vector store. When a query is received, the system retrieves the most relevant chunks from the vector store and uses a large language model (LLM) to generate an answer.

### Knowledge Graph-based RAG

This method uses a knowledge graph to represent the entities and relationships in the financial documents. The `LightRAG` library is used to build the knowledge graph and perform retrieval. This approach can provide more precise and contextually aware answers by leveraging the structured information in the knowledge graph.