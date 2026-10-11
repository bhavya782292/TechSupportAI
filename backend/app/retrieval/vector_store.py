import chromadb

from backend.app.core.config import PROJECT_ROOT


CHROMA_PATH = PROJECT_ROOT / "data" / "indexes" / "chroma"


class VectorStore:
    """Manage the persistent ChromaDB vector collection."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name="technical_support_chunks"
        )

    def add_documents(
        self,
        documents: list,
        embeddings: list[list[float]],
    ) -> None:
        """Store child documents, embeddings, and metadata in ChromaDB."""

        ids = [
            f"{document.metadata['source']}:{document.metadata['chunk_id']}"
            for document in documents
        ]

        self.collection.upsert(
            ids=ids,
            documents=[document.page_content for document in documents],
            embeddings=embeddings,
            metadatas=[document.metadata for document in documents],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Search ChromaDB and return documents with retrieval metadata."""

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_documents = []

        for index, document in enumerate(results["documents"][0]):
            retrieved_documents.append(
                {
                    "id": results["ids"][0][index],
                    "document": document,
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                }
            )

        return retrieved_documents


vector_store = VectorStore()