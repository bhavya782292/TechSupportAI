from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for a technical-support query."""

    question: str = Field(
        ...,
        min_length=1,
        description="Technical support question from the user.",
    )


class QueryResponse(BaseModel):
    """Response returned by the RAG API."""

    answer: str