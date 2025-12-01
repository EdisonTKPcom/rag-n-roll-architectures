"""
Naive RAG - Basic Retrieval-Augmented Generation

Architecture:
    Query → Vector Retriever → Top-K Context → LLM → Response

This is the simplest RAG pattern where we:
1. Embed the user query
2. Retrieve top-K similar documents from a vector store
3. Pass the context + query to an LLM for generation

Usage:
    python 1_naive_rag.py
"""

import numpy as np
from typing import List, Dict, Any


# =============================================================================
# MOCK COMPONENTS (Replace with real implementations for production)
# =============================================================================

class MockEmbedding:
    """Mock embedding model - replace with OpenAI, Sentence Transformers, etc."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        """Generate a mock embedding based on text hash."""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dim).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(t) for t in texts])


class MockVectorStore:
    """Mock vector store - replace with ChromaDB, Faiss, Pinecone, etc."""
    
    def __init__(self, embedding_model: MockEmbedding):
        self.embedding = embedding_model
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: List[str]):
        """Index documents into the vector store."""
        self.documents.extend(documents)
        new_embeddings = self.embedding.embed_batch(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        print(f"[VectorStore] Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        query_embedding = self.embedding.embed(query)
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx])
            })
        return results


class MockLLM:
    """Mock LLM - replace with OpenAI, Anthropic, local models, etc."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response based on the prompt."""
        # In production, this would call an actual LLM API
        return f"[MockLLM Response] Based on the provided context, I can help answer your question. " \
               f"The context contains relevant information that addresses the query."


# =============================================================================
# NAIVE RAG IMPLEMENTATION
# =============================================================================

class NaiveRAG:
    """
    Naive RAG: Simple retrieval-augmented generation.
    
    Flow:
        1. User submits a query
        2. Query is embedded and used to retrieve top-K similar documents
        3. Retrieved documents are combined as context
        4. Context + query are passed to LLM for response generation
    """
    
    def __init__(self, top_k: int = 3):
        self.embedding = MockEmbedding()
        self.vector_store = MockVectorStore(self.embedding)
        self.llm = MockLLM()
        self.top_k = top_k
    
    def index_documents(self, documents: List[str]):
        """Add documents to the knowledge base."""
        self.vector_store.add_documents(documents)
    
    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Build the prompt with retrieved context."""
        context_text = "\n\n".join([
            f"[Document {i+1}] (score: {doc['score']:.3f})\n{doc['document']}"
            for i, doc in enumerate(context)
        ])
        
        prompt = f"""Use the following context to answer the question.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        return prompt
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing the response and retrieved context
        """
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)
        
        # Step 1: Retrieve relevant documents
        print("\n[Step 1] Retrieving relevant documents...")
        retrieved_docs = self.vector_store.search(question, top_k=self.top_k)
        for i, doc in enumerate(retrieved_docs):
            print(f"  Doc {i+1} (score: {doc['score']:.3f}): {doc['document'][:50]}...")
        
        # Step 2: Build prompt with context
        print("\n[Step 2] Building prompt with context...")
        prompt = self._build_prompt(question, retrieved_docs)
        
        # Step 3: Generate response
        print("\n[Step 3] Generating response...")
        response = self.llm.generate(prompt)
        print(f"  Response: {response}")
        
        return {
            "question": question,
            "response": response,
            "retrieved_documents": retrieved_docs,
            "prompt": prompt
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    print("=" * 60)
    print("NAIVE RAG EXAMPLE")
    print("Architecture: Query → Vector Retriever → Top-K Context → LLM → Response")
    print("=" * 60)
    
    # Initialize RAG system
    rag = NaiveRAG(top_k=3)
    
    # Sample knowledge base
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "RAG (Retrieval-Augmented Generation) combines retrieval with generative AI models.",
        "Vector databases store embeddings for efficient similarity search.",
        "LLMs (Large Language Models) are trained on vast amounts of text data.",
        "Transformers are the architecture behind most modern language models.",
        "Embeddings are dense vector representations of text or other data.",
        "Semantic search finds results based on meaning rather than keywords.",
    ]
    
    # Index documents
    print("\nIndexing documents...")
    rag.index_documents(documents)
    
    # Run test queries
    test_queries = [
        "What is RAG?",
        "How do vector databases work?",
        "Tell me about Python programming",
    ]
    
    for query in test_queries:
        result = rag.query(query)
    
    print("\n" + "=" * 60)
    print("NAIVE RAG EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
