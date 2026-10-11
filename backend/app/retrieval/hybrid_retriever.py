from backend.app.retrieval.bm25_store import bm25_store
from backend.app.retrieval.embedding_service import embedding_service
from backend.app.retrieval.vector_store import vector_store


class HybridRetriever:
    """Combine dense and BM25 retrieval using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve and fuse results from dense and BM25 search."""

        query_embedding = embedding_service.embed_text(query)

        dense_results = vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        bm25_results = bm25_store.search(
            query,
            top_k=top_k,
        )

        fused_results: dict[str, dict] = {}

        for rank, result in enumerate(dense_results, start=1):
            document_id = result["id"]

            if document_id not in fused_results:
                fused_results[document_id] = {
                    "id": document_id,
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                }

            fused_results[document_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        for rank, result in enumerate(bm25_results, start=1):
            document_id = result["id"]

            if document_id not in fused_results:
                fused_results[document_id] = {
                    "id": document_id,
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                }

            fused_results[document_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        return sorted(
            fused_results.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )[:top_k]


hybrid_retriever = HybridRetriever()