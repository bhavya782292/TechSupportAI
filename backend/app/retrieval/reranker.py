from sentence_transformers import CrossEncoder




class Reranker:
    """Rerank retrieved documents using a Cross-Encoder model."""

    def __init__(self) -> None:
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score query-document pairs and return reranked results."""

        if not documents:
            return []

        pairs = [
            (query, document["document"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked_documents = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["reranker_score"] = float(score)
            reranked_documents.append(result)

        reranked_documents.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return reranked_documents[:top_k]


reranker = Reranker()