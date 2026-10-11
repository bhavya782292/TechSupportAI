from rank_bm25 import BM25Okapi


class BM25Store:
    """Manage sparse keyword-based retrieval using BM25."""

    def __init__(self) -> None:
        self.documents: list = []
        self.bm25: BM25Okapi | None = None

    def build_index(self, documents: list) -> None:
        """Build the BM25 index from child documents."""

        self.documents = documents

        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search BM25 and return documents with scores and metadata."""

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built. Call build_index() first."
            )

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[index]

            results.append(
                {
                    "id": (
                        f"{document.metadata['source']}:"
                        f"{document.metadata['chunk_id']}"
                    ),
                    "document": document.page_content,
                    "metadata": document.metadata,
                    "score": float(scores[index]),
                }
            )

        return results


bm25_store = BM25Store()