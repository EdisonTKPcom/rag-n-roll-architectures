"""
Graph RAG

Architecture:
    Documents -.ingest.-> Graph DB
                            ↓
    Query → Graph/KG Query → Graph DB → Subgraph/Triples
                                              ↓
                                    LLM Graph Explainer → Grounded Answer

This pattern uses a knowledge graph for retrieval:
1. Documents are ingested and converted to a knowledge graph
2. Queries are transformed into graph queries
3. Relevant subgraphs or triples are retrieved
4. LLM explains and answers based on graph structure

Usage:
    python 4_graph_rag.py
"""

from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class Triple:
    """A knowledge graph triple (subject, predicate, object)."""
    subject: str
    predicate: str
    obj: str  # 'object' is a reserved word
    
    def __str__(self):
        return f"({self.subject}) --[{self.predicate}]--> ({self.obj})"
    
    def __hash__(self):
        return hash((self.subject, self.predicate, self.obj))
    
    def __eq__(self, other):
        return (self.subject, self.predicate, self.obj) == (other.subject, other.predicate, other.obj)


@dataclass
class Entity:
    """A knowledge graph entity with properties."""
    name: str
    entity_type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


# =============================================================================
# MOCK COMPONENTS
# =============================================================================

class MockGraphDB:
    """
    Mock knowledge graph database.
    
    In production, use:
    - Neo4j
    - Amazon Neptune
    - JanusGraph
    - Microsoft GraphRAG
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.triples: List[Triple] = []
        self.adjacency: Dict[str, List[Triple]] = defaultdict(list)
    
    def add_entity(self, entity: Entity):
        """Add an entity to the graph."""
        self.entities[entity.name] = entity
    
    def add_triple(self, triple: Triple):
        """Add a relationship triple."""
        self.triples.append(triple)
        self.adjacency[triple.subject].append(triple)
        self.adjacency[triple.obj].append(triple)
    
    def get_entity(self, name: str) -> Entity:
        """Get entity by name."""
        return self.entities.get(name)
    
    def get_neighbors(self, entity_name: str, hops: int = 1) -> Set[Triple]:
        """Get all triples within N hops of an entity."""
        visited = set()
        frontier = {entity_name}
        result_triples = set()
        
        for _ in range(hops):
            next_frontier = set()
            for entity in frontier:
                if entity in visited:
                    continue
                visited.add(entity)
                for triple in self.adjacency[entity]:
                    result_triples.add(triple)
                    next_frontier.add(triple.subject)
                    next_frontier.add(triple.obj)
            frontier = next_frontier - visited
        
        return result_triples
    
    def query_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of a given type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def query_by_predicate(self, predicate: str) -> List[Triple]:
        """Find all triples with a given predicate."""
        return [t for t in self.triples if t.predicate == predicate]


class MockEntityExtractor:
    """
    Mock entity and relationship extractor.
    
    In production, use:
    - spaCy NER
    - OpenAI function calling
    - Microsoft GraphRAG extractor
    """
    
    def extract(self, text: str) -> Tuple[List[Entity], List[Triple]]:
        """Extract entities and relationships from text."""
        # Simple rule-based extraction for demo
        entities = []
        triples = []
        
        # Mock extraction logic
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entities.append(Entity(word.strip(".,"), "ENTITY"))
        
        return entities, triples


class MockQueryTransformer:
    """Transforms natural language to graph queries."""
    
    def __init__(self, graph_db: MockGraphDB):
        self.graph_db = graph_db
    
    def extract_entities(self, query: str) -> List[str]:
        """Find entities mentioned in the query."""
        found = []
        query_lower = query.lower()
        for entity_name in self.graph_db.entities:
            if entity_name.lower() in query_lower:
                found.append(entity_name)
        return found
    
    def transform(self, query: str) -> Dict[str, Any]:
        """Transform query into graph operations."""
        entities = self.extract_entities(query)
        return {
            "entities": entities,
            "operation": "get_subgraph",
            "hops": 2
        }


class MockGraphExplainer:
    """LLM that explains graph context and generates answers."""
    
    def explain(self, query: str, subgraph: Set[Triple], entities: List[Entity]) -> str:
        """Generate an explanation based on graph context."""
        triple_strs = [str(t) for t in subgraph]
        entity_strs = [f"{e.name} ({e.entity_type})" for e in entities]
        
        return (
            f"[MockGraphExplainer] Based on the knowledge graph:\n"
            f"  Entities found: {entity_strs}\n"
            f"  Relationships: {triple_strs[:5]}{'...' if len(triple_strs) > 5 else ''}\n"
            f"  This grounded response is traceable back to specific graph connections."
        )


# =============================================================================
# GRAPH RAG IMPLEMENTATION
# =============================================================================

class GraphRAG:
    """
    Graph RAG: Uses knowledge graphs for structured retrieval.
    
    Flow:
        1. Ingest documents → Extract entities & relationships → Build graph
        2. Transform query → Identify entities → Retrieve subgraph
        3. LLM explains answer using graph context
    
    Benefits:
        - Explainable: answers trace back to specific relationships
        - Structured: captures complex relationships between entities
        - Reasoning: enables multi-hop reasoning over connections
    """
    
    def __init__(self):
        self.graph_db = MockGraphDB()
        self.extractor = MockEntityExtractor()
        self.query_transformer = MockQueryTransformer(self.graph_db)
        self.explainer = MockGraphExplainer()
    
    def ingest_document(self, text: str):
        """Extract and add entities/relationships from a document."""
        entities, triples = self.extractor.extract(text)
        for entity in entities:
            self.graph_db.add_entity(entity)
        for triple in triples:
            self.graph_db.add_triple(triple)
    
    def add_triple(self, subject: str, predicate: str, obj: str, 
                   subj_type: str = "ENTITY", obj_type: str = "ENTITY"):
        """Manually add a triple to the graph."""
        self.graph_db.add_entity(Entity(subject, subj_type))
        self.graph_db.add_entity(Entity(obj, obj_type))
        self.graph_db.add_triple(Triple(subject, predicate, obj))
    
    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)
        
        # Step 1: Transform query to graph operations
        print("\n[Step 1] Transforming query to graph operations...")
        graph_query = self.query_transformer.transform(question)
        print(f"  Found entities: {graph_query['entities']}")
        
        # Step 2: Retrieve relevant subgraph
        print("\n[Step 2] Retrieving subgraph from knowledge graph...")
        subgraph = set()
        entities = []
        
        for entity_name in graph_query["entities"]:
            entity = self.graph_db.get_entity(entity_name)
            if entity:
                entities.append(entity)
                entity_triples = self.graph_db.get_neighbors(entity_name, hops=graph_query["hops"])
                subgraph.update(entity_triples)
        
        print(f"  Retrieved {len(subgraph)} triples for {len(entities)} entities")
        for triple in list(subgraph)[:5]:
            print(f"    {triple}")
        
        # Step 3: Generate grounded explanation
        print("\n[Step 3] Generating grounded explanation...")
        response = self.explainer.explain(question, subgraph, entities)
        print(f"  {response}")
        
        return {
            "question": question,
            "response": response,
            "subgraph": subgraph,
            "entities": entities,
            "graph_query": graph_query
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    print("=" * 60)
    print("GRAPH RAG EXAMPLE")
    print("Architecture: Query → Graph Query → KG → Subgraph → LLM Explainer")
    print("=" * 60)
    
    # Initialize
    rag = GraphRAG()
    
    # Build a sample knowledge graph
    print("\nBuilding knowledge graph...")
    
    # People and their roles
    rag.add_triple("Einstein", "was_a", "Physicist", "PERSON", "PROFESSION")
    rag.add_triple("Einstein", "developed", "Theory of Relativity", "PERSON", "THEORY")
    rag.add_triple("Einstein", "born_in", "Germany", "PERSON", "COUNTRY")
    rag.add_triple("Einstein", "worked_at", "Princeton", "PERSON", "INSTITUTION")
    
    # Related concepts
    rag.add_triple("Theory of Relativity", "explains", "Spacetime", "THEORY", "CONCEPT")
    rag.add_triple("Theory of Relativity", "includes", "E=mc²", "THEORY", "EQUATION")
    rag.add_triple("E=mc²", "relates", "Mass-Energy Equivalence", "EQUATION", "CONCEPT")
    
    # More people
    rag.add_triple("Newton", "was_a", "Physicist", "PERSON", "PROFESSION")
    rag.add_triple("Newton", "developed", "Laws of Motion", "PERSON", "THEORY")
    rag.add_triple("Newton", "born_in", "England", "PERSON", "COUNTRY")
    
    # Connections
    rag.add_triple("Theory of Relativity", "superseded", "Newtonian Mechanics", "THEORY", "THEORY")
    rag.add_triple("Laws of Motion", "part_of", "Newtonian Mechanics", "THEORY", "THEORY")
    
    print(f"  Graph has {len(rag.graph_db.entities)} entities and {len(rag.graph_db.triples)} triples")
    
    # Test queries
    test_queries = [
        "What did Einstein develop?",
        "Tell me about the Theory of Relativity",
        "How are Einstein and Newton related?",
    ]
    
    for query in test_queries:
        result = rag.query(query)
    
    print("\n" + "=" * 60)
    print("GRAPH RAG EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
