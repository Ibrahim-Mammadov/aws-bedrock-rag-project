# AWS Bedrock RAG System

Production-grade Retrieval-Augmented Generation (RAG) pipeline built with Amazon Bedrock, LangChain, and FAISS. This system enables context-aware question answering over custom private documents by combining semantic vector search with foundational large language models.

## Architecture & Workflow

The architecture follows a standard RAG pattern, decoupling document ingestion/indexing from runtime retrieval and generation:

1. **Ingestion & Text Splitting**: Raw documents are loaded and split into semantically manageable chunks using recursive character splitting.
2. **Vector Embeddings**: Text chunks are transformed into high-dimensional vector representations using Amazon Titan Embeddings (`amazon.titan-embed-text-v1`).
3. **Vector Indexing**: Embeddings are indexed and stored locally using FAISS (Facebook AI Similarity Search) for rapid nearest-neighbor retrieval.
4. **Contextual Retrieval**: User queries trigger a similarity search against the vector store to fetch the top-k most relevant document contexts.
5. **LLM Generation**: The retrieved context and user prompt are fed into the Amazon Nova model (`eu.amazon.nova-lite-v1:0`) via Amazon Bedrock Runtime API to synthesize an accurate, source-grounded response.

```
[Document] -> [Text Splitter] -> [Titan Embeddings] -> [FAISS Vector Store]
                                                              |
[User Query] -> [Similarity Search] -> [Context + Query] -> [Amazon Nova Model] -> [Answer]
```

## Tech Stack

- **Cloud Provider**: Amazon Web Services (AWS Bedrock)
- **Orchestration Framework**: LangChain (`langchain`, `langchain-community`, `langchain-aws`)
- **Vector Database**: FAISS (`faiss-cpu`)
- **Language Models**: 
  - Embeddings: `amazon.titan-embed-text-v1`
  - Generation: `eu.amazon.nova-lite-v1:0` (Amazon Nova Lite)
- **Language**: Python 3.10+

## Project Structure

```
aws-bedrock-rag-project/
├── .gitignore          # Excludes virtual environments, caches, and local vector stores
├── requirements.txt    # Python package dependencies
├── document.txt        # Source text document for ingestion
├── create_database.py  # Script for chunking, embedding, and building the FAISS index
├── query_rag.py        # Script for semantic retrieval and LLM generation
└── test_bedrock.py     # Connection verification script for AWS Bedrock
```

## Prerequisites

- AWS Account with active access to Amazon Bedrock models (`Amazon Titan Embeddings` and `Amazon Nova`).
- AWS CLI configured locally with appropriate IAM permissions (`boto3` credentials chain).
- Python 3.10 or higher.

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ibrahim-Mammadov/aws-bedrock-rag-project.git
   cd aws-bedrock-rag-project
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify AWS Connection**:
   ```bash
   python test_bedrock.py
   ```

## Execution Guide

### Step 1: Create the Vector Database
Process the source text (`document.txt`), generate vector embeddings via Amazon Bedrock, and persist the index locally:
```bash
python create_database.py
```

### Step 2: Query the RAG Pipeline
Execute a semantic similarity search and generate a grounded response using Amazon Nova:
```bash
python query_rag.py
```

## Security & Best Practices

- **Credential Isolation**: Never hardcode AWS credentials. Utilize AWS IAM roles, environment variables, or standard AWS credentials files (`~/.aws/credentials`).
- **Data Privacy**: Vector indexes and sensitive source documents are excluded from version control via `.gitignore`.
- **Model Governance**: Employs managed foundation endpoints through AWS Bedrock with enterprise-grade security compliance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
