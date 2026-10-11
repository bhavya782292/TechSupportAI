
from langchain_core.documents import Document
from langsmith import traceable

from backend.app.retrieval.hybrid_retriever import hybrid_retriever
from backend.app.retrieval.parent_child_retriever import (
    parent_child_retriever,
)
from backend.app.retrieval.reranker import reranker


class RetrievalPipeline:
    """Coordinate hybrid retrieval, reranking, and parent lookup."""

    @traceable(
        name="TechSupportAI_Retrieval_Pipeline",
        run_type="chain",
    )
    def retrieve(
        self,
        query: str,
        parent_documents: list[Document],
        candidate_k: int = 10,
        final_k: int = 5,
    ) -> list[dict]:
        """Run hybrid retrieval, reranking, and parent lookup."""

        parent_child_retriever.build_parent_index(parent_documents)

        candidates = hybrid_retriever.retrieve(
            query,
            top_k=candidate_k,
        )

        reranked_results = reranker.rerank(
            query,
            candidates,
            top_k=final_k,
        )

        final_results = []

        for child in reranked_results:
            parent_id = child["metadata"]["parent_id"]
            parent = parent_child_retriever.get_parent(parent_id)

            if parent is None:
                continue

            final_results.append(
                {
                    "child_id": child["id"],
                    "child_document": child["document"],
                    "child_metadata": child["metadata"],
                    "rrf_score": child["rrf_score"],
                    "reranker_score": child["reranker_score"],
                    "parent_id": parent_id,
                    "parent_document": parent.page_content,
                }
            )

        return final_results


retrieval_pipeline = RetrievalPipeline()
