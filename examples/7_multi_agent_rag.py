"""
Multi-Agent RAG

Architecture:
    Query → Coordinator Agent
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
      Research   Retrieval  Reasoning
       Agent      Agent      Agent
          ↓         ↓         ↓
       Web/APIs  Vector/SQL  Tools/Calc
          └─────────┼─────────┘
                    ↓
           Evidence Merge + Critique
                    ↓
             Writer Agent → Final Answer + Sources

This pattern uses multiple specialized agents:
1. Coordinator delegates to specialized agents
2. Each agent has its own tools and capabilities
3. Evidence is merged and critiqued
4. Writer agent synthesizes final response

Usage:
    python 7_multi_agent_rag.py
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    content: Any
    message_type: str = "data"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """A piece of evidence with source attribution."""
    content: str
    source: str
    confidence: float
    agent: str


@dataclass
class CritiqueResult:
    """Result of evidence critique."""
    original: List[Evidence]
    verified: List[Evidence]
    conflicts: List[str]
    summary: str


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.messages: List[AgentMessage] = []
    
    def receive(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.messages.append(message)
    
    @abstractmethod
    def process(self, task: str) -> AgentMessage:
        """Process a task and return result."""
        pass


# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================

class ResearchAgent(BaseAgent):
    """
    Research Agent: Handles web and API queries.
    
    Specializes in:
    - Web search
    - API calls
    - External data gathering
    """
    
    def __init__(self):
        super().__init__("ResearchAgent")
        self.web_sources = [
            {"title": "Wikipedia", "type": "encyclopedia"},
            {"title": "ArXiv", "type": "papers"},
            {"title": "News API", "type": "news"},
        ]
    
    def process(self, task: str) -> AgentMessage:
        print(f"  [ResearchAgent] Processing: '{task}'")
        
        # Simulate web research
        evidence = []
        if "python" in task.lower():
            evidence.append(Evidence(
                content="Python is a high-level programming language created in 1991.",
                source="Wikipedia",
                confidence=0.95,
                agent=self.name
            ))
        if "ai" in task.lower() or "machine learning" in task.lower():
            evidence.append(Evidence(
                content="Latest AI research focuses on large language models and multimodal systems.",
                source="ArXiv",
                confidence=0.85,
                agent=self.name
            ))
        
        # Default research result
        if not evidence:
            evidence.append(Evidence(
                content=f"Research findings related to: {task}",
                source="Web Search",
                confidence=0.7,
                agent=self.name
            ))
        
        return AgentMessage(
            sender=self.name,
            content=evidence,
            message_type="evidence"
        )


class RetrievalAgent(BaseAgent):
    """
    Retrieval Agent: Handles internal document and data retrieval.
    
    Specializes in:
    - Vector search
    - Graph queries
    - SQL/structured data
    """
    
    def __init__(self):
        super().__init__("RetrievalAgent")
        self.documents = [
            "Internal doc: Company was founded in 2010 with a focus on enterprise software.",
            "Technical spec: The system uses a microservices architecture with REST APIs.",
            "Policy: All code must pass security review before deployment.",
        ]
    
    def _vector_search(self, query: str) -> List[str]:
        """Simulate vector search."""
        results = []
        query_words = set(query.lower().split())
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            if query_words & doc_words:
                results.append(doc)
        return results if results else [self.documents[0]]
    
    def process(self, task: str) -> AgentMessage:
        print(f"  [RetrievalAgent] Processing: '{task}'")
        
        # Perform retrieval
        retrieved = self._vector_search(task)
        
        evidence = [
            Evidence(
                content=doc,
                source="Vector Store",
                confidence=0.8,
                agent=self.name
            )
            for doc in retrieved
        ]
        
        return AgentMessage(
            sender=self.name,
            content=evidence,
            message_type="evidence"
        )


class ReasoningAgent(BaseAgent):
    """
    Reasoning/Tooling Agent: Handles calculations and logical operations.
    
    Specializes in:
    - Mathematical calculations
    - Logical reasoning
    - Tool execution
    """
    
    def __init__(self):
        super().__init__("ReasoningAgent")
        self.tools = {
            "calculator": self._calculate,
            "date_parser": self._parse_date,
            "unit_converter": self._convert_units,
        }
    
    def _calculate(self, expression: str) -> str:
        """Simple calculator."""
        try:
            # Very basic - only for demo
            if "+" in expression:
                parts = expression.split("+")
                return str(sum(float(p.strip()) for p in parts))
            return "Calculation result"
        except Exception:
            return "Could not calculate"
    
    def _parse_date(self, date_str: str) -> str:
        return f"Parsed date: {date_str}"
    
    def _convert_units(self, value: str) -> str:
        return f"Converted: {value}"
    
    def process(self, task: str) -> AgentMessage:
        print(f"  [ReasoningAgent] Processing: '{task}'")
        
        evidence = []
        
        # Check for calculation needs
        if any(op in task for op in ["+", "-", "*", "/", "calculate", "compute"]):
            result = self._calculate(task)
            evidence.append(Evidence(
                content=f"Calculation result: {result}",
                source="Calculator Tool",
                confidence=1.0,
                agent=self.name
            ))
        
        # Logical reasoning
        evidence.append(Evidence(
            content=f"Reasoning analysis of: {task}",
            source="Reasoning Engine",
            confidence=0.75,
            agent=self.name
        ))
        
        return AgentMessage(
            sender=self.name,
            content=evidence,
            message_type="evidence"
        )


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent: Orchestrates other agents.
    
    Responsibilities:
    - Task decomposition
    - Agent delegation
    - Result aggregation
    """
    
    def __init__(self, agents: List[BaseAgent]):
        super().__init__("Coordinator")
        self.agents = {agent.name: agent for agent in agents}
    
    def delegate(self, task: str) -> Dict[str, AgentMessage]:
        """Delegate task to all specialized agents."""
        print(f"\n[Coordinator] Delegating task: '{task}'")
        
        results = {}
        for name, agent in self.agents.items():
            if name != self.name:
                result = agent.process(task)
                results[name] = result
        
        return results
    
    def process(self, task: str) -> AgentMessage:
        results = self.delegate(task)
        return AgentMessage(
            sender=self.name,
            content=results,
            message_type="delegation_results"
        )


class CritiqueAgent(BaseAgent):
    """
    Critique Agent: Merges and validates evidence.
    
    Responsibilities:
    - Evidence deduplication
    - Conflict detection
    - Confidence weighting
    """
    
    def __init__(self):
        super().__init__("CritiqueAgent")
    
    def critique(self, all_evidence: List[Evidence]) -> CritiqueResult:
        """Critique and merge evidence from multiple agents."""
        print(f"\n[CritiqueAgent] Critiquing {len(all_evidence)} pieces of evidence...")
        
        # Simple deduplication by content similarity
        verified = []
        seen_content = set()
        
        for ev in sorted(all_evidence, key=lambda x: x.confidence, reverse=True):
            content_key = ev.content[:50].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                verified.append(ev)
        
        # Detect potential conflicts (simplified)
        conflicts = []
        
        return CritiqueResult(
            original=all_evidence,
            verified=verified,
            conflicts=conflicts,
            summary=f"Verified {len(verified)} of {len(all_evidence)} evidence pieces"
        )
    
    def process(self, task: str) -> AgentMessage:
        # This agent is called with evidence, not raw task
        return AgentMessage(
            sender=self.name,
            content="Critique complete",
            message_type="critique"
        )


class WriterAgent(BaseAgent):
    """
    Writer Agent: Synthesizes final response.
    
    Responsibilities:
    - Response generation
    - Source citation
    - Formatting
    """
    
    def __init__(self):
        super().__init__("WriterAgent")
    
    def write(self, query: str, critique_result: CritiqueResult) -> str:
        """Write the final response with citations."""
        print(f"\n[WriterAgent] Writing final response...")
        
        # Build response with citations
        evidence_summary = "\n".join([
            f"  - [{ev.source}] ({ev.confidence:.0%}): {ev.content[:60]}..."
            for ev in critique_result.verified[:5]
        ])
        
        response = (
            f"[WriterAgent Final Response]\n"
            f"Based on evidence from {len(critique_result.verified)} verified sources:\n\n"
            f"Summary: The multi-agent system has gathered and validated information "
            f"to answer your query.\n\n"
            f"Key Evidence:\n{evidence_summary}\n\n"
            f"Sources: {set(ev.source for ev in critique_result.verified)}"
        )
        
        return response
    
    def process(self, task: str) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            content=task,
            message_type="response"
        )


# =============================================================================
# MULTI-AGENT RAG IMPLEMENTATION
# =============================================================================

class MultiAgentRAG:
    """
    Multi-Agent RAG: Orchestrates multiple specialized agents.
    
    Flow:
        1. Coordinator receives query
        2. Delegates to Research, Retrieval, and Reasoning agents
        3. Critique agent merges and validates evidence
        4. Writer agent produces final response with citations
    
    Benefits:
        - Specialization: each agent is expert in its domain
        - Validation: evidence is critiqued before use
        - Comprehensive: multiple perspectives combined
        - Traceable: sources are cited
    """
    
    def __init__(self):
        # Create specialized agents
        self.research_agent = ResearchAgent()
        self.retrieval_agent = RetrievalAgent()
        self.reasoning_agent = ReasoningAgent()
        
        # Create orchestration agents
        self.coordinator = CoordinatorAgent([
            self.research_agent,
            self.retrieval_agent,
            self.reasoning_agent
        ])
        self.critique_agent = CritiqueAgent()
        self.writer_agent = WriterAgent()
    
    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)
        
        # Step 1: Coordinator delegates to agents
        print("\n[Step 1] Coordinator delegating to specialized agents...")
        delegation_results = self.coordinator.delegate(question)
        
        # Step 2: Collect all evidence
        print("\n[Step 2] Collecting evidence from all agents...")
        all_evidence: List[Evidence] = []
        for agent_name, message in delegation_results.items():
            if message.message_type == "evidence":
                evidence_list = message.content
                all_evidence.extend(evidence_list)
                print(f"  {agent_name}: {len(evidence_list)} pieces of evidence")
        
        # Step 3: Critique and merge evidence
        print("\n[Step 3] Critiquing and merging evidence...")
        critique_result = self.critique_agent.critique(all_evidence)
        print(f"  {critique_result.summary}")
        
        # Step 4: Writer produces final response
        print("\n[Step 4] Writer producing final response...")
        final_response = self.writer_agent.write(question, critique_result)
        print(f"\n{final_response}")
        
        return {
            "question": question,
            "delegation_results": delegation_results,
            "all_evidence": all_evidence,
            "critique_result": critique_result,
            "response": final_response
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    print("=" * 60)
    print("MULTI-AGENT RAG EXAMPLE")
    print("Architecture: Coordinator → [Research + Retrieval + Reasoning] → Critique → Writer")
    print("=" * 60)
    
    # Initialize
    rag = MultiAgentRAG()
    
    # Test queries
    test_queries = [
        "Tell me about Python and its uses in AI",
        "What is the company's security policy?",
        "Calculate 100 + 200 and explain the result",
    ]
    
    for query in test_queries:
        result = rag.query(query)
    
    print("\n" + "=" * 60)
    print("MULTI-AGENT RAG EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
