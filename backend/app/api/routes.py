from fastapi import APIRouter

from backend.app.generation.rag_generator import rag_generator
from backend.app.models.schemas import QueryRequest, QueryResponse


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_support(request: QueryRequest) -> QueryResponse:
    """Answer a technical-support question using the RAG pipeline."""

    answer = rag_generator.generate(request.question)

    return QueryResponse(answer=answer)