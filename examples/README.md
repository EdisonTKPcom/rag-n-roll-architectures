# RAG Examples - Runnable Code

This directory contains runnable Python examples for each RAG architecture pattern documented in the main repository.

## Overview

Each example implements a basic, runnable version of the corresponding RAG pattern using mock components. These examples are designed to:

1. **Demonstrate the architecture** - Show how components connect and data flows
2. **Be runnable** - Execute without external dependencies (using mock LLMs, vector stores, etc.)
3. **Be extensible** - Easy to swap mock components with real implementations

## Examples

| # | Pattern | File | Description |
|---|---------|------|-------------|
| 1 | Naive RAG | `1_naive_rag.py` | Simple query → retrieve → generate pipeline |
| 2 | Rerank RAG | `2_rerank_rag.py` | Adds cross-encoder reranking for better relevance |
| 3 | Multimodal RAG | `3_multimodal_rag.py` | Handles text, image, and audio modalities |
| 4 | Graph RAG | `4_graph_rag.py` | Uses knowledge graphs for structured retrieval |
| 5 | Hybrid RAG | `5_hybrid_rag.py` | Combines vector, keyword, and graph search |
| 6 | Agentic RAG | `6_agent_router_rag.py` | Agent routes to appropriate tools/sources |
| 7 | Multi-Agent RAG | `7_multi_agent_rag.py` | Multiple specialized agents collaborate |

## Quick Start

### Prerequisites

- Python 3.8+
- NumPy (only required dependency)

### Installation

```bash
# Install minimal dependencies
pip install numpy

# Or install all optional dependencies for extending examples
pip install -r requirements.txt
```

### Running Examples

Each example is self-contained and can be run directly:

```bash
# Run a specific example
python 1_naive_rag.py

# Run all examples
for i in {1..7}; do python ${i}_*.py; done
```

## Example Output

Running `1_naive_rag.py`:

```
============================================================
NAIVE RAG EXAMPLE
Architecture: Query → Vector Retriever → Top-K Context → LLM → Response
============================================================

Indexing documents...
[VectorStore] Indexed 8 documents

============================================================
Query: What is RAG?
============================================================

[Step 1] Retrieving relevant documents...
  Doc 1 (score: 0.XXX): RAG (Retrieval-Augmented Generation) combines...
  Doc 2 (score: 0.XXX): Vector databases store embeddings...
  Doc 3 (score: 0.XXX): LLMs (Large Language Models) are trained...

[Step 2] Building prompt with context...

[Step 3] Generating response...
  Response: [MockLLM Response] Based on the provided context...

============================================================
NAIVE RAG EXAMPLE COMPLETED SUCCESSFULLY!
============================================================
```

## Extending Examples

### Using Real LLMs

Replace `MockLLM` with real implementations:

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

class RealLLM:
    def generate(self, prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

### Using Real Vector Stores

Replace `MockVectorStore` with real implementations:

```python
# ChromaDB
import chromadb

class RealVectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("docs")
    
    def add_documents(self, documents):
        self.collection.add(
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
    
    def search(self, query, top_k=3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results
```

### Using Real Embeddings

Replace `MockEmbedding` with real models:

```python
# Sentence Transformers
from sentence_transformers import SentenceTransformer

class RealEmbedding:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text):
        return self.model.encode(text)
```

## Architecture Patterns

### 1. Naive RAG
```
Query → Embed → Vector Search → Top-K → Prompt → LLM → Response
```

### 2. Rerank RAG
```
Query → Vector Search → Many Candidates → Reranker → Top-K → LLM → Response
```

### 3. Multimodal RAG
```
Query → Modality Router → [Text|Image|Audio] Retrievers → Fusion → MLLM → Response
```

### 4. Graph RAG
```
Query → Entity Extraction → Graph Traversal → Subgraph → LLM Explainer → Response
```

### 5. Hybrid RAG
```
Query → [Vector + BM25 + Graph] → RRF Fusion → Dedup → LLM → Response
```

### 6. Agentic RAG
```
Query → Tool Router → [Web|DB|Docs|Apps] → Context Aggregation → Planner → LLM
```

### 7. Multi-Agent RAG
```
Query → Coordinator → [Research|Retrieval|Reasoning] → Critique → Writer → Response
```

## Contributing

Feel free to:
- Add more examples (e.g., Hierarchical RAG, Memory-Augmented RAG)
- Improve mock implementations
- Add real integration examples
- Fix bugs or improve documentation

## License

MIT - Same as the parent repository.
