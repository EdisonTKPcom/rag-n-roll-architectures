# RAG‑n‑Roll: Architectures You Can Jam With 🎸

A friendly, visual tour of modern Retrieval‑Augmented Generation (RAG) patterns — from Naive to Multi‑Agent.  
Each diagram is written in **Mermaid**, so GitHub will render it natively.

## Patterns covered
1. Naive RAG
2. Retrieve‑and‑Rerank RAG
3. Multimodal RAG
4. Graph RAG
5. Hybrid RAG
6. Agentic RAG (Router)
7. Multi‑Agent RAG

---

## Quick Glance Diagrams

> Click any heading to open the dedicated diagram file.

### 1) Naive RAG
[View diagram](diagrams/1_naive_rag.md)
```mermaid
flowchart LR
    Q[Query] --> RET[Vector Retriever]
    RET --> C[Top‑K Context]
    C --> LLM[LLM]
    LLM --> R[Response]
```

### 2) Retrieve‑and‑Rerank RAG
[View diagram](diagrams/2_rerank_rag.md)
```mermaid
flowchart LR
    Q[Query] --> RET[Vector Retriever]
    RET --> C[Candidate Chunks]
    Q --> RR[Reranker]
    C --> RR
    RR --> C2[Re‑ranked Context]
    C2 --> LLM[LLM]
    LLM --> R[Response]
```

### 3) Multimodal RAG
[View diagram](diagrams/3_multimodal_rag.md)
```mermaid
flowchart LR
    Q[Query (Text/Img/Audio)] --> ROUTE{Modality Router}
    ROUTE -->|Text| RET_T[Text Retriever]
    ROUTE -->|Image| RET_I[Image/Vector Retriever]
    ROUTE -->|Audio/Video| RET_A[AV Index Retriever]
    RET_T --> CT[Text Context]
    RET_I --> CI[Image/Embed Context]
    RET_A --> CA[AV Context]
    CT & CI & CA --> FUSE[Context Fusion]
    FUSE --> MLLM[Multimodal LLM]
    MLLM --> R[Response]
```

### 4) Graph RAG
[View diagram](diagrams/4_graph_rag.md)
```mermaid
flowchart LR
    Q[Query] --> KGR[Graph/KG Query]
    DOCS[Documents] -.ingest.-> KG[(Graph DB)]
    KGR --> KG
    KG --> SUBG[Subgraph / Triples]
    SUBG --> EXP[LLM Graph Explainer]
    EXP --> R[Grounded, Explainable Answer]
```

### 5) Hybrid RAG
[View diagram](diagrams/5_hybrid_rag.md)
```mermaid
flowchart LR
    Q[Query] --> VRET[Vector Search]
    Q --> KRET[Keyword/BM25]
    Q --> GRET[Graph/Structured]
    VRET --> VCTX[Vector Matches]
    KRET --> KCTX[Keyword Matches]
    GRET --> GCTX[Graph Facts]
    VCTX & KCTX & GCTX --> FUSE[Hybrid Fusion + Dedup]
    FUSE --> LLM[LLM]
    LLM --> R[Response]
```

### 6) Agentic RAG (Router)
[View diagram](diagrams/6_agent_router_rag.md)
```mermaid
flowchart LR
    Q[User Query] --> AGENT{Policy/Tool Router}
    AGENT -->|Web| WEB[Web Search Tool]
    AGENT -->|DB| DB[SQL/BI Tool]
    AGENT -->|Docs| VRET[Vector Retriever]
    AGENT -->|Email/Apps| APPS[Connectors]
    WEB & DB & VRET & APPS --> CTX[Unified Context]
    CTX --> PLAN[Agent Planner]
    PLAN --> LLM[LLM]
    LLM --> R[Response]
```

### 7) Multi‑Agent RAG
[View diagram](diagrams/7_multi_agent_rag.md)
```mermaid
flowchart LR
    Q[Query] --> CORD[Coordinator Agent]
    CORD --> R1[Research Agent]
    CORD --> R2[Retrieval Agent]
    CORD --> R3[Reasoning/Tooling Agent]
    R1 --> K1[Web / APIs]
    R2 --> K2[Vector / Graph / SQL]
    R3 --> K3[Calculators / Tools]
    K1 & K2 & K3 --> CMB[Evidence Merge + Critique]
    CMB --> WRITER[Writer Agent -> LLM]
    WRITER --> R[Final Answer + Sources]
```

---

## How to use
- Open any `diagrams/*.md` file to view a specific pattern.
- Copy a block into your own README, wiki, or slide deck.
- PRs welcome: add variants (e.g., **Hierarchical RAG**, **Query Expansion**, **Memory‑Augmented RAG**).

## License
MIT
