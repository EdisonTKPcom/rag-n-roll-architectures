"""
Hybrid RAG

Architecture:
    Query → ┌─ Vector Search ────→ Vector Matches ─┐
            ├─ Keyword/BM25 ─────→ Keyword Matches ─├→ Hybrid Fusion + Dedup → LLM → Response
            └─ Graph/Structured ─→ Graph Facts ────┘

This pattern combines multiple retrieval methods:
1. Vector/semantic search for meaning-based matching
2. Keyword/BM25 for exact term matching
3. Graph/structured queries for factual data
4. Fusion layer to combine and deduplicate results

Usage:
    python 5_hybrid_rag.py
"""

import numpy as np
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import re


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class Document:
    """A document with ID and content."""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


@dataclass
class RetrievalResult:
    """Result from a retriever with source info."""
    document: Document
    score: float
    source: str  # "vector", "keyword", "graph"


# =============================================================================
# MOCK RETRIEVERS
# =============================================================================

class MockVectorRetriever:
    """Semantic vector search retriever."""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
    
    def _embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).astype(np.float32)
    
    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        embeddings = np.array([self._embed(d.content) for d in documents])
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        query_emb = self._embed(query)
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            RetrievalResult(
                document=self.documents[i],
                score=float(similarities[i]),
                source="vector"
            )
            for i in top_indices
        ]


class MockBM25Retriever:
    """Keyword-based BM25 retriever."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.documents: List[Document] = []
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_terms: List[Dict[str, int]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    def add_documents(self, documents: List[Document]):
        for doc in documents:
            self.documents.append(doc)
            terms = self._tokenize(doc.content)
            term_freq = defaultdict(int)
            for term in terms:
                term_freq[term] += 1
            self.doc_terms.append(dict(term_freq))
            
            for term in set(terms):
                self.doc_freqs[term] += 1
        
        total_len = sum(len(self._tokenize(d.content)) for d in self.documents)
        self.avg_doc_len = total_len / len(self.documents) if self.documents else 0
    
    def _bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_terms = self.doc_terms[doc_idx]
        doc_len = sum(doc_terms.values())
        n_docs = len(self.documents)
        
        for term in query_terms:
            if term not in doc_terms:
                continue
            
            tf = doc_terms[term]
            df = self.doc_freqs.get(term, 0)
            
            # IDF
            idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            
            # TF normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
            )
            
            score += idf * tf_norm
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        scores = [(i, self._bm25_score(query_terms, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(
                document=self.documents[i],
                score=score,
                source="keyword"
            )
            for i, score in scores[:top_k] if score > 0
        ]


class MockStructuredRetriever:
    """Structured/graph-based retriever for factual queries."""
    
    def __init__(self):
        self.facts: Dict[str, List[Document]] = defaultdict(list)
    
    def add_documents(self, documents: List[Document]):
        """Index documents by their metadata tags."""
        for doc in documents:
            for key, value in doc.metadata.items():
                self.facts[f"{key}:{value}"].append(doc)
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for documents matching structured criteria in query."""
        results = []
        query_lower = query.lower()
        
        for key, docs in self.facts.items():
            tag_key, tag_value = key.split(":", 1)
            if tag_value.lower() in query_lower:
                for doc in docs:
                    results.append(RetrievalResult(
                        document=doc,
                        score=1.0,
                        source="graph"
                    ))
        
        return results[:top_k]


# =============================================================================
# FUSION LAYER
# =============================================================================

class HybridFusion:
    """Fuses and deduplicates results from multiple retrievers."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "vector": 0.4,
            "keyword": 0.3,
            "graph": 0.3
        }
    
    def reciprocal_rank_fusion(self, 
                                results: Dict[str, List[RetrievalResult]], 
                                k: int = 60) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) for combining ranked lists.
        
        RRF score = sum(1 / (k + rank)) for each list where document appears
        """
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, RetrievalResult] = {}
        
        for source, source_results in results.items():
            weight = self.weights.get(source, 1.0)
            for rank, result in enumerate(source_results, 1):
                doc_id = result.document.id
                doc_scores[doc_id] += weight / (k + rank)
                
                # Keep the result with highest individual score
                if doc_id not in doc_map or result.score > doc_map[doc_id].score:
                    doc_map[doc_id] = result
        
        # Sort by fused score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        return [
            RetrievalResult(
                document=doc_map[doc_id].document,
                score=doc_scores[doc_id],
                source="hybrid"
            )
            for doc_id in sorted_ids
        ]
    
    def deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate documents, keeping highest scored."""
        seen: Set[str] = set()
        deduped = []
        for result in results:
            if result.document.id not in seen:
                seen.add(result.document.id)
                deduped.append(result)
        return deduped


class MockLLM:
    """Mock LLM for response generation."""
    
    def generate(self, prompt: str) -> str:
        return (
            "[MockLLM Response] Based on hybrid retrieval combining semantic similarity, "
            "keyword matching, and structured data, here is the comprehensive answer."
        )


# =============================================================================
# HYBRID RAG IMPLEMENTATION
# =============================================================================

class HybridRAG:
    """
    Hybrid RAG: Combines multiple retrieval strategies.
    
    Flow:
        1. Query all retrievers in parallel
        2. Fuse results using Reciprocal Rank Fusion
        3. Deduplicate and rank
        4. Generate response with combined context
    
    Benefits:
        - Best of both worlds: semantic + keyword + structured
        - Robust: handles different query types effectively
        - Comprehensive: gathers evidence from multiple sources
    """
    
    def __init__(self, top_k: int = 5):
        self.vector_retriever = MockVectorRetriever()
        self.bm25_retriever = MockBM25Retriever()
        self.structured_retriever = MockStructuredRetriever()
        self.fusion = HybridFusion()
        self.llm = MockLLM()
        self.top_k = top_k
    
    def index_documents(self, documents: List[Document]):
        """Index documents across all retrievers."""
        self.vector_retriever.add_documents(documents)
        self.bm25_retriever.add_documents(documents)
        self.structured_retriever.add_documents(documents)
        print(f"[HybridRAG] Indexed {len(documents)} documents across all retrievers")
    
    def _build_prompt(self, query: str, results: List[RetrievalResult]) -> str:
        context = "\n\n".join([
            f"[{r.source}] (score: {r.score:.3f})\n{r.document.content}"
            for r in results
        ])
        return f"""Answer based on the following hybrid-retrieved context:

{context}

Question: {query}

Answer:"""
    
    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)
        
        # Step 1: Query all retrievers
        print("\n[Step 1] Querying all retrievers in parallel...")
        
        vector_results = self.vector_retriever.search(question, self.top_k)
        print(f"  Vector: {len(vector_results)} results")
        
        keyword_results = self.bm25_retriever.search(question, self.top_k)
        print(f"  Keyword/BM25: {len(keyword_results)} results")
        
        structured_results = self.structured_retriever.search(question, self.top_k)
        print(f"  Structured: {len(structured_results)} results")
        
        # Step 2: Fuse results
        print("\n[Step 2] Fusing results with Reciprocal Rank Fusion...")
        all_results = {
            "vector": vector_results,
            "keyword": keyword_results,
            "graph": structured_results
        }
        fused = self.fusion.reciprocal_rank_fusion(all_results)
        deduped = self.fusion.deduplicate(fused)[:self.top_k]
        print(f"  Fused and deduped: {len(deduped)} unique documents")
        
        for i, r in enumerate(deduped[:3]):
            print(f"    {i+1}. [{r.source}] (score: {r.score:.3f}): {r.document.content[:40]}...")
        
        # Step 3: Generate response
        print("\n[Step 3] Generating response...")
        prompt = self._build_prompt(question, deduped)
        response = self.llm.generate(prompt)
        print(f"  {response}")
        
        return {
            "question": question,
            "response": response,
            "vector_results": vector_results,
            "keyword_results": keyword_results,
            "structured_results": structured_results,
            "fused_results": deduped
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    print("=" * 60)
    print("HYBRID RAG EXAMPLE")
    print("Architecture: Query → [Vector + BM25 + Graph] → Fusion → LLM")
    print("=" * 60)
    
    # Initialize
    rag = HybridRAG(top_k=5)
    
    # Sample documents with metadata
    documents = [
        Document("1", "Python is a versatile programming language for web and data science.",
                {"topic": "programming", "language": "python"}),
        Document("2", "The Python programming language was created by Guido van Rossum in 1991.",
                {"topic": "history", "language": "python"}),
        Document("3", "Machine learning with Python uses libraries like scikit-learn and TensorFlow.",
                {"topic": "ml", "language": "python"}),
        Document("4", "JavaScript is the language of the web browser.",
                {"topic": "programming", "language": "javascript"}),
        Document("5", "RAG systems combine retrieval with language model generation.",
                {"topic": "ai", "category": "rag"}),
        Document("6", "Vector databases store embeddings for semantic similarity search.",
                {"topic": "databases", "category": "rag"}),
        Document("7", "BM25 is a probabilistic ranking function used in information retrieval.",
                {"topic": "ir", "category": "search"}),
        Document("8", "Hybrid search combines keyword and semantic matching for better results.",
                {"topic": "search", "category": "rag"}),
    ]
    
    print("\nIndexing documents...")
    rag.index_documents(documents)
    
    # Test queries
    test_queries = [
        "What is Python used for?",  # Good for both vector and keyword
        "Tell me about RAG systems",  # Good for structured + semantic
        "How does BM25 ranking work?",  # Good for keyword match
    ]
    
    for query in test_queries:
        result = rag.query(query)
    
    print("\n" + "=" * 60)
    print("HYBRID RAG EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
