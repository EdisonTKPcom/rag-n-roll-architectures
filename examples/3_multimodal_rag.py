"""
Multimodal RAG

Architecture:
    Query (Text/Img/Audio) → Modality Router
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Text Retriever  Image Retriever  AV Retriever
                    ↓               ↓               ↓
                Text Context  Image Context   AV Context
                    └───────────────┼───────────────┘
                                    ↓
                            Context Fusion
                                    ↓
                            Multimodal LLM → Response

This pattern handles multiple input modalities:
1. Route query based on content type (text, image, audio)
2. Retrieve from modality-specific indexes
3. Fuse multimodal context
4. Generate response with a multimodal LLM

Usage:
    python 3_multimodal_rag.py
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# DATA TYPES
# =============================================================================

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class MultimodalQuery:
    """A query that can contain multiple modalities."""
    text: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    
    def get_modalities(self) -> List[Modality]:
        """Return list of present modalities."""
        modalities = []
        if self.text:
            modalities.append(Modality.TEXT)
        if self.image_path:
            modalities.append(Modality.IMAGE)
        if self.audio_path:
            modalities.append(Modality.AUDIO)
        return modalities


@dataclass
class MultimodalDocument:
    """A document that can contain multiple modalities."""
    id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# MOCK COMPONENTS
# =============================================================================

class MockTextRetriever:
    """Mock text retriever using embeddings."""
    
    def __init__(self):
        self.documents: List[MultimodalDocument] = []
        self.embeddings: np.ndarray = None
    
    def _embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).astype(np.float32)
    
    def add_documents(self, documents: List[MultimodalDocument]):
        text_docs = [d for d in documents if d.text]
        self.documents.extend(text_docs)
        embeddings = np.array([self._embed(d.text) for d in text_docs])
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            return []
        query_emb = self._embed(query)
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [{"document": self.documents[i], "score": float(similarities[i])} for i in top_indices]


class MockImageRetriever:
    """Mock image retriever using CLIP-like embeddings."""
    
    def __init__(self):
        self.documents: List[MultimodalDocument] = []
        self.embeddings: np.ndarray = None
    
    def _embed_image(self, image_path: str) -> np.ndarray:
        """Simulate CLIP image embedding."""
        np.random.seed(hash(image_path) % (2**32))
        return np.random.randn(512).astype(np.float32)
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Simulate CLIP text embedding for cross-modal search."""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(512).astype(np.float32)
    
    def add_documents(self, documents: List[MultimodalDocument]):
        image_docs = [d for d in documents if d.image_path]
        self.documents.extend(image_docs)
        embeddings = np.array([self._embed_image(d.image_path) for d in image_docs])
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search images using text query (cross-modal)."""
        if self.embeddings is None:
            return []
        query_emb = self._embed_text(query)
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [{"document": self.documents[i], "score": float(similarities[i])} for i in top_indices]


class MockAudioRetriever:
    """Mock audio/video retriever."""
    
    def __init__(self):
        self.documents: List[MultimodalDocument] = []
    
    def add_documents(self, documents: List[MultimodalDocument]):
        audio_docs = [d for d in documents if d.audio_path]
        self.documents.extend(audio_docs)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search audio documents."""
        # Simulate simple keyword matching for mock
        results = []
        for doc in self.documents:
            if doc.text and any(word in doc.text.lower() for word in query.lower().split()):
                results.append({"document": doc, "score": 0.8})
        return results[:top_k]


class MockModalityRouter:
    """Routes queries to appropriate retrievers based on modality."""
    
    def route(self, query: MultimodalQuery) -> List[Modality]:
        """Determine which retrievers to use."""
        modalities = query.get_modalities()
        print(f"[Router] Detected modalities: {[m.value for m in modalities]}")
        return modalities if modalities else [Modality.TEXT]


class MockContextFusion:
    """Fuses context from multiple modalities."""
    
    def fuse(self, contexts: Dict[Modality, List[Dict[str, Any]]]) -> str:
        """Combine multimodal contexts into a unified representation."""
        fused_parts = []
        
        for modality, docs in contexts.items():
            if docs:
                fused_parts.append(f"\n=== {modality.value.upper()} CONTEXT ===")
                for i, doc in enumerate(docs):
                    doc_obj = doc["document"]
                    fused_parts.append(f"[{modality.value}_{i+1}] {doc_obj.text or doc_obj.image_path or doc_obj.audio_path}")
        
        return "\n".join(fused_parts) if fused_parts else "No context found."


class MockMultimodalLLM:
    """Mock multimodal LLM (like GPT-4V, Gemini, etc.)."""
    
    def generate(self, query: MultimodalQuery, context: str) -> str:
        modalities = query.get_modalities()
        return f"[MockMultimodalLLM] Analyzed {len(modalities)} modality(ies) with fused context. " \
               f"This response integrates information from all available sources."


# =============================================================================
# MULTIMODAL RAG IMPLEMENTATION
# =============================================================================

class MultimodalRAG:
    """
    Multimodal RAG: Handles text, image, and audio inputs.
    
    Flow:
        1. Route query based on modality content
        2. Retrieve from modality-specific indexes
        3. Fuse multimodal context
        4. Generate with multimodal LLM
    
    Use cases:
        - Visual question answering
        - Audio/video search and summarization
        - Cross-modal retrieval (text → image, image → text)
    """
    
    def __init__(self, top_k: int = 3):
        self.router = MockModalityRouter()
        self.text_retriever = MockTextRetriever()
        self.image_retriever = MockImageRetriever()
        self.audio_retriever = MockAudioRetriever()
        self.fusion = MockContextFusion()
        self.llm = MockMultimodalLLM()
        self.top_k = top_k
    
    def index_documents(self, documents: List[MultimodalDocument]):
        """Index documents across all modality retrievers."""
        self.text_retriever.add_documents(documents)
        self.image_retriever.add_documents(documents)
        self.audio_retriever.add_documents(documents)
        print(f"[MultimodalRAG] Indexed {len(documents)} documents")
    
    def query(self, query: MultimodalQuery) -> Dict[str, Any]:
        query_text = query.text or "[non-text query]"
        print(f"\n{'='*60}")
        print(f"Query: {query_text}")
        print('='*60)
        
        # Step 1: Route to appropriate retrievers
        print("\n[Step 1] Routing query to modality-specific retrievers...")
        modalities = self.router.route(query)
        
        # Step 2: Retrieve from each modality
        print("\n[Step 2] Retrieving from each modality...")
        contexts: Dict[Modality, List[Dict[str, Any]]] = {}
        
        search_query = query.text or ""
        
        if Modality.TEXT in modalities or search_query:
            contexts[Modality.TEXT] = self.text_retriever.search(search_query, self.top_k)
            print(f"  Text: Found {len(contexts[Modality.TEXT])} results")
        
        if Modality.IMAGE in modalities or search_query:
            contexts[Modality.IMAGE] = self.image_retriever.search(search_query, self.top_k)
            print(f"  Image: Found {len(contexts[Modality.IMAGE])} results")
        
        if Modality.AUDIO in modalities:
            contexts[Modality.AUDIO] = self.audio_retriever.search(search_query, self.top_k)
            print(f"  Audio: Found {len(contexts[Modality.AUDIO])} results")
        
        # Step 3: Fuse context
        print("\n[Step 3] Fusing multimodal context...")
        fused_context = self.fusion.fuse(contexts)
        print(f"  Fused context preview: {fused_context[:100]}...")
        
        # Step 4: Generate response
        print("\n[Step 4] Generating multimodal response...")
        response = self.llm.generate(query, fused_context)
        print(f"  Response: {response}")
        
        return {
            "query": query,
            "response": response,
            "contexts": contexts,
            "fused_context": fused_context
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    print("=" * 60)
    print("MULTIMODAL RAG EXAMPLE")
    print("Architecture: Query → Modality Router → Multi-Retrievers → Fusion → MLLM")
    print("=" * 60)
    
    # Initialize
    rag = MultimodalRAG(top_k=2)
    
    # Sample multimodal knowledge base
    documents = [
        MultimodalDocument(id="1", text="The Eiffel Tower is a wrought-iron lattice tower in Paris, France."),
        MultimodalDocument(id="2", text="Machine learning models can classify images with high accuracy.", 
                          image_path="/images/ml_diagram.png"),
        MultimodalDocument(id="3", text="Neural networks are inspired by biological brain structures.",
                          image_path="/images/neural_net.png"),
        MultimodalDocument(id="4", text="Podcast episode discussing the future of AI technology.",
                          audio_path="/audio/ai_podcast.mp3"),
        MultimodalDocument(id="5", text="Paris has many famous landmarks including the Louvre Museum.",
                          image_path="/images/paris_landmarks.jpg"),
        MultimodalDocument(id="6", text="Voice assistants use speech recognition and NLU.",
                          audio_path="/audio/voice_tech.mp3"),
    ]
    
    print("\nIndexing multimodal documents...")
    rag.index_documents(documents)
    
    # Test queries
    test_queries = [
        # Text-only query
        MultimodalQuery(text="Tell me about landmarks in Paris"),
        # Text query about images
        MultimodalQuery(text="Show me diagrams about neural networks"),
        # Query with image reference
        MultimodalQuery(text="What is this?", image_path="/query/unknown_image.jpg"),
    ]
    
    for query in test_queries:
        result = rag.query(query)
    
    print("\n" + "=" * 60)
    print("MULTIMODAL RAG EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
