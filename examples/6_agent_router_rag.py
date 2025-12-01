"""
Agentic RAG (Router)

Architecture:
    User Query → Policy/Tool Router
                        │
          ┌─────────────┼─────────────┬─────────────┐
          ↓             ↓             ↓             ↓
      Web Search   SQL/BI Tool   Vector Retriever  Connectors
          └─────────────┼─────────────┴─────────────┘
                        ↓
                 Unified Context
                        ↓
                  Agent Planner → LLM → Response

This pattern uses an agent to route queries to appropriate tools:
1. Agent analyzes the query intent
2. Routes to appropriate data sources (web, DB, docs, apps)
3. Aggregates results into unified context
4. Plans and generates comprehensive response

Usage:
    python 6_agent_router_rag.py
"""

import numpy as np
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# DATA TYPES
# =============================================================================

class ToolType(Enum):
    WEB_SEARCH = "web"
    DATABASE = "db"
    VECTOR_DOCS = "docs"
    APPS = "apps"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool: str
    data: Any
    success: bool
    error: str = None


@dataclass
class RoutingDecision:
    """Decision about which tools to use."""
    tools: List[ToolType]
    reasoning: str
    query_rewrite: str = None


# =============================================================================
# MOCK TOOLS
# =============================================================================

class MockWebSearchTool:
    """Mock web search tool - simulates searching the internet."""
    
    def search(self, query: str, num_results: int = 3) -> ToolResult:
        print(f"  [WebSearch] Searching: '{query}'")
        # Simulate web search results
        results = [
            {"title": f"Web Result {i+1}", "snippet": f"Information about {query}...", "url": f"https://example.com/{i}"}
            for i in range(num_results)
        ]
        return ToolResult(tool="web_search", data=results, success=True)


class MockDatabaseTool:
    """Mock SQL/BI tool for structured data queries."""
    
    def __init__(self):
        # Simulate a simple database
        self.data = {
            "sales": [
                {"product": "Widget A", "revenue": 150000, "quarter": "Q1"},
                {"product": "Widget B", "revenue": 200000, "quarter": "Q1"},
                {"product": "Widget A", "revenue": 175000, "quarter": "Q2"},
            ],
            "users": [
                {"name": "Alice", "department": "Engineering", "projects": 5},
                {"name": "Bob", "department": "Sales", "projects": 3},
            ]
        }
    
    def query(self, sql_like_query: str) -> ToolResult:
        print(f"  [Database] Executing: '{sql_like_query}'")
        # Simple mock - return relevant data based on keywords
        if "sales" in sql_like_query.lower() or "revenue" in sql_like_query.lower():
            return ToolResult(tool="database", data=self.data["sales"], success=True)
        elif "user" in sql_like_query.lower() or "employee" in sql_like_query.lower():
            return ToolResult(tool="database", data=self.data["users"], success=True)
        return ToolResult(tool="database", data=[], success=True)


class MockVectorRetrieverTool:
    """Mock vector retriever for document search."""
    
    def __init__(self):
        self.documents = [
            "Company policy: All employees must complete annual security training.",
            "Technical documentation: The API uses OAuth2 for authentication.",
            "Product guide: Widget A is our flagship product for enterprise customers.",
            "Meeting notes: Q2 planning discussed expanding to new markets.",
        ]
    
    def search(self, query: str, top_k: int = 3) -> ToolResult:
        print(f"  [VectorDocs] Searching: '{query}'")
        # Simple keyword-based mock
        results = []
        query_words = set(query.lower().split())
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            if query_words & doc_words:
                results.append({"content": doc, "score": len(query_words & doc_words) / len(query_words)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return ToolResult(tool="vector_docs", data=results[:top_k], success=True)


class MockAppConnectorTool:
    """Mock connector for external apps (email, calendar, etc.)."""
    
    def __init__(self):
        self.emails = [
            {"from": "ceo@company.com", "subject": "Q3 Goals", "snippet": "Our focus for Q3 is..."},
            {"from": "hr@company.com", "subject": "Benefits Update", "snippet": "New benefits starting..."},
        ]
        self.calendar = [
            {"title": "Team Standup", "time": "9:00 AM", "recurring": True},
            {"title": "Product Review", "time": "2:00 PM", "recurring": False},
        ]
    
    def get_data(self, app: str, query: str) -> ToolResult:
        print(f"  [Apps] Fetching from {app}: '{query}'")
        if "email" in app.lower():
            return ToolResult(tool=f"apps:{app}", data=self.emails, success=True)
        elif "calendar" in app.lower():
            return ToolResult(tool=f"apps:{app}", data=self.calendar, success=True)
        return ToolResult(tool=f"apps:{app}", data=[], success=True)


# =============================================================================
# AGENT COMPONENTS
# =============================================================================

class MockPolicyRouter:
    """
    Routes queries to appropriate tools based on intent.
    
    In production, this could be:
    - LLM-based function calling
    - Trained classifier
    - Rule-based routing
    """
    
    def __init__(self):
        # Keywords that suggest certain tools
        self.tool_keywords = {
            ToolType.WEB_SEARCH: ["latest", "news", "current", "today", "search", "find online"],
            ToolType.DATABASE: ["sales", "revenue", "data", "statistics", "numbers", "query", "sql", "report"],
            ToolType.VECTOR_DOCS: ["policy", "documentation", "guide", "how to", "procedure", "internal"],
            ToolType.APPS: ["email", "calendar", "meeting", "schedule", "message"],
        }
    
    def route(self, query: str) -> RoutingDecision:
        """Determine which tools to use for the query."""
        query_lower = query.lower()
        selected_tools = []
        
        for tool_type, keywords in self.tool_keywords.items():
            if any(kw in query_lower for kw in keywords):
                selected_tools.append(tool_type)
        
        # Default to vector docs if no specific tool matched
        if not selected_tools:
            selected_tools = [ToolType.VECTOR_DOCS]
        
        reasoning = f"Query contains keywords suggesting: {[t.value for t in selected_tools]}"
        
        return RoutingDecision(
            tools=selected_tools,
            reasoning=reasoning,
            query_rewrite=query  # Could rewrite for each tool
        )


class MockAgentPlanner:
    """Plans how to use tool results to answer the query."""
    
    def plan(self, query: str, tool_results: List[ToolResult]) -> str:
        """Create an execution plan based on available results."""
        available_data = [r.tool for r in tool_results if r.success and r.data]
        return f"Plan: Synthesize answer using data from {available_data}"


class MockAgentLLM:
    """LLM for final response generation."""
    
    def generate(self, query: str, context: str, plan: str) -> str:
        return (
            f"[MockAgentLLM] Based on the routing decision and aggregated context "
            f"from multiple tools, here is the comprehensive answer to your query."
        )


# =============================================================================
# AGENTIC RAG IMPLEMENTATION
# =============================================================================

class AgenticRAG:
    """
    Agentic RAG: Uses an agent to route and orchestrate retrieval.
    
    Flow:
        1. Policy router analyzes query intent
        2. Routes to appropriate tools (web, DB, docs, apps)
        3. Aggregates results into unified context
        4. Agent planner creates response strategy
        5. LLM generates final response
    
    Benefits:
        - Flexible: adapts to different query types
        - Comprehensive: can pull from multiple sources
        - Intelligent: uses reasoning to select tools
    """
    
    def __init__(self):
        self.router = MockPolicyRouter()
        self.web_tool = MockWebSearchTool()
        self.db_tool = MockDatabaseTool()
        self.docs_tool = MockVectorRetrieverTool()
        self.apps_tool = MockAppConnectorTool()
        self.planner = MockAgentPlanner()
        self.llm = MockAgentLLM()
    
    def _execute_tool(self, tool_type: ToolType, query: str) -> ToolResult:
        """Execute a specific tool."""
        if tool_type == ToolType.WEB_SEARCH:
            return self.web_tool.search(query)
        elif tool_type == ToolType.DATABASE:
            return self.db_tool.query(query)
        elif tool_type == ToolType.VECTOR_DOCS:
            return self.docs_tool.search(query)
        elif tool_type == ToolType.APPS:
            return self.apps_tool.get_data("email", query)
        return ToolResult(tool="unknown", data=None, success=False, error="Unknown tool")
    
    def _format_context(self, results: List[ToolResult]) -> str:
        """Format tool results into unified context."""
        parts = []
        for result in results:
            if result.success and result.data:
                parts.append(f"\n=== {result.tool.upper()} ===")
                if isinstance(result.data, list):
                    for item in result.data[:3]:  # Limit items
                        parts.append(f"  - {item}")
                else:
                    parts.append(f"  {result.data}")
        return "\n".join(parts) if parts else "No relevant data found."
    
    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)
        
        # Step 1: Route query to tools
        print("\n[Step 1] Routing query to appropriate tools...")
        routing = self.router.route(question)
        print(f"  Selected tools: {[t.value for t in routing.tools]}")
        print(f"  Reasoning: {routing.reasoning}")
        
        # Step 2: Execute selected tools
        print("\n[Step 2] Executing tools...")
        tool_results = []
        for tool_type in routing.tools:
            result = self._execute_tool(tool_type, question)
            tool_results.append(result)
            status = "✓" if result.success else "✗"
            print(f"  {status} {tool_type.value}: {len(result.data) if result.data else 0} results")
        
        # Step 3: Aggregate context
        print("\n[Step 3] Aggregating unified context...")
        context = self._format_context(tool_results)
        print(f"  Context preview: {context[:100]}...")
        
        # Step 4: Plan response
        print("\n[Step 4] Planning response...")
        plan = self.planner.plan(question, tool_results)
        print(f"  {plan}")
        
        # Step 5: Generate response
        print("\n[Step 5] Generating response...")
        response = self.llm.generate(question, context, plan)
        print(f"  {response}")
        
        return {
            "question": question,
            "routing": routing,
            "tool_results": tool_results,
            "context": context,
            "plan": plan,
            "response": response
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    print("=" * 60)
    print("AGENTIC RAG (ROUTER) EXAMPLE")
    print("Architecture: Query → Router → [Tools] → Planner → LLM")
    print("=" * 60)
    
    # Initialize
    rag = AgenticRAG()
    
    # Test different query types
    test_queries = [
        "What are the latest sales numbers for Q1?",  # DB + possibly web
        "Find the company policy on security training",  # Vector docs
        "Check my emails about Q3 goals",  # Apps
        "Search online for the latest AI trends",  # Web search
        "How do I authenticate with the API?",  # Vector docs (technical)
    ]
    
    for query in test_queries:
        result = rag.query(query)
    
    print("\n" + "=" * 60)
    print("AGENTIC RAG EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
