```mermaid
flowchart LR
    Q[Query] --> RET[Vector Retriever]
    RET --> C[Top‑K Context]
    C --> LLM[LLM]
    LLM --> R[Response]
```
