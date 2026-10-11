
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings


class EmbeddingService:
    """Generate semantic embeddings for text using Sentence Transformers."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_text(self, text: str) -> list[float]:
        """Convert a single text into a 384-dimensional embedding vector."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into embedding vectors."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()


embedding_service = EmbeddingService()
