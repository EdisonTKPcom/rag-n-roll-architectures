"""
Retrieve-and-Rerank RAG

Architecture:
    Query → Vector Retriever → Candidate Chunks
                                      ↓
    Query → Reranker ← Candidate Chunks
                  ↓
           Re-ranked Context → LLM → Response

This pattern improves upon Naive RAG by adding a reranking step:
1. Retrieve a larger set of candidate documents
2. Use a cross-encoder reranker to score query-document pairs
3. Select top documents after reranking for better relevance

Usage:
    python 2_rerank_rag.py
"""

import numpy as np
from typing import List, Dict, Any


# =============================================================================
# MOCK COMPONENTS
# =============================================================================

class MockEmbedding:
    """Mock embedding model."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dim).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class MockVectorStore:
    """Mock vector store."""
    
    def __init__(self, embedding_model: MockEmbedding):
        self.embedding = embedding_model
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: List[str]):
        self.documents.extend(documents)
        new_embeddings = self.embedding.embed_batch(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        print(f"[VectorStore] Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embedding.embed(query)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx])
            })
        return results


class MockReranker:
    """
    Mock reranker - simulates a cross-encoder model.
    
    In production, use models like:
    - Cohere Rerank
    - sentence-transformers/ms-marco-MiniLM-L-6-v2
    - BGE Reranker
    """
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Cross-encoders score query-document pairs jointly, which is more
        accurate than bi-encoder similarity but slower.
        """
        print(f"[Reranker] Reranking {len(documents)} candidates...")
        
        reranked = []
        for doc in documents:
            # Simulate cross-encoder scoring based on word overlap
            query_words = set(query.lower().split())
            doc_words = set(doc["document"].lower().split())
            overlap = len(query_words & doc_words)
            
            # Combine original score with overlap for mock reranking
            rerank_score = doc["score"] * 0.3 + (overlap / max(len(query_words), 1)) * 0.7
            
            reranked.append({
                "document": doc["document"],
                "original_score": doc["score"],
                "rerank_score": rerank_score
            })
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


class MockLLM:
    """Mock LLM."""
    
    def generate(self, prompt: str) -> str:
        return f"[MockLLM Response] Based on the reranked context, here is my answer. " \
               f"The reranking step ensures the most relevant documents are prioritized."


# =============================================================================
# RERANK RAG IMPLEMENTATION
# =============================================================================

class RerankRAG:
    """
    Retrieve-and-Rerank RAG: Improves retrieval quality with reranking.
    
    Flow:
        1. Retrieve a larger set of candidate documents (e.g., top-20)
        2. Rerank candidates using a cross-encoder model
        3. Select top-K after reranking
        4. Generate response with reranked context
    
    Benefits:
        - Better relevance than pure vector search
        - Cross-encoders can capture nuanced query-document relationships
        - Reduces noise in the context
    """
    
    def __init__(self, retrieve_k: int = 10, rerank_k: int = 3):
        self.embedding = MockEmbedding()
        self.vector_store = MockVectorStore(self.embedding)
        self.reranker = MockReranker()
        self.llm = MockLLM()
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k
    
    def index_documents(self, documents: List[str]):
        self.vector_store.add_documents(documents)
    
    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([
            f"[Document {i+1}] (rerank score: {doc['rerank_score']:.3f})\n{doc['document']}"
            for i, doc in enumerate(context)
        ])
        
        return f"""Use the following reranked context to answer the question.

CONTEXT (reranked by relevance):
{context_text}

QUESTION: {query}

ANSWER:"""
    
    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)
        
        # Step 1: Initial retrieval (get more candidates)
        print(f"\n[Step 1] Initial retrieval (top-{self.retrieve_k} candidates)...")
        candidates = self.vector_store.search(question, top_k=self.retrieve_k)
        for i, doc in enumerate(candidates[:5]):  # Show first 5
            print(f"  Candidate {i+1} (score: {doc['score']:.3f}): {doc['document'][:40]}...")
        
        # Step 2: Rerank candidates
        print(f"\n[Step 2] Reranking to get top-{self.rerank_k}...")
        reranked_docs = self.reranker.rerank(question, candidates, top_k=self.rerank_k)
        for i, doc in enumerate(reranked_docs):
            print(f"  Reranked {i+1} (score: {doc['rerank_score']:.3f}): {doc['document'][:40]}...")
        
        # Step 3: Build prompt and generate
        print("\n[Step 3] Generating response with reranked context...")
        prompt = self._build_prompt(question, reranked_docs)
        response = self.llm.generate(prompt)
        print(f"  Response: {response}")
        
        return {
            "question": question,
            "response": response,
            "candidates": candidates,
            "reranked_documents": reranked_docs
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    print("=" * 60)
    print("RETRIEVE-AND-RERANK RAG EXAMPLE")
    print("Architecture: Query → Retriever → Candidates → Reranker → LLM → Response")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RerankRAG(retrieve_k=8, rerank_k=3)
    
    # Sample knowledge base
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Python snakes are non-venomous constrictors found in Asia and Africa.",
        "Monty Python was a British comedy group famous for absurdist humor.",
        "Machine learning uses Python extensively for data science applications.",
        "RAG combines retrieval with generation for grounded AI responses.",
        "Vector databases enable efficient semantic similarity search.",
        "Cross-encoders score query-document pairs for better ranking.",
        "Bi-encoders embed queries and documents separately for efficiency.",
        "Reranking improves precision by rescoring retrieved candidates.",
        "LLMs benefit from relevant context to reduce hallucinations.",
    ]
    
    print("\nIndexing documents...")
    rag.index_documents(documents)
    
    # Test queries
    test_queries = [
        "What programming language is good for machine learning?",
        "How does reranking improve RAG systems?",
    ]
    
    for query in test_queries:
        result = rag.query(query)
    
    print("\n" + "=" * 60)
    print("RETRIEVE-AND-RERANK RAG EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
