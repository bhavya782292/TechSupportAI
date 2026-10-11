from langchain_core.documents import Document

from backend.app.retrieval.hybrid_retriever import hybrid_retriever


class ParentChildRetriever:
    """Retrieve precise child chunks and map them to parent documents."""

    def __init__(self) -> None:
        self.parent_documents: dict[str, Document] = {}

    def build_parent_index(
        self,
        parent_documents: list[Document],
    ) -> None:
        """Create a lookup table from parent ID to parent document."""

        self.parent_documents = {
            document.metadata["source"]: document
            for document in parent_documents
        }

    def get_parent(
        self,
        parent_id: str,
    ) -> Document | None:
        """Return the parent document for a given parent ID."""

        return self.parent_documents.get(parent_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve child chunks and attach their parent documents."""

        child_results = hybrid_retriever.retrieve(
            query,
            top_k=top_k,
        )

        results = []

        for child in child_results:
            parent_id = child["metadata"]["parent_id"]

            parent = self.get_parent(parent_id)

            if parent is None:
                continue

            results.append(
                {
                    "child_id": child["id"],
                    "child_document": child["document"],
                    "child_metadata": child["metadata"],
                    "rrf_score": child["rrf_score"],
                    "parent_id": parent_id,
                    "parent_document": parent.page_content,
                }
            )

        return results


parent_child_retriever = ParentChildRetriever()