# Multimodal RAG
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
