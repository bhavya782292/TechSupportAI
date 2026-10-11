from fastapi import FastAPI

from backend.app.api.routes import router


app = FastAPI(
    title="TechSupportAI",
    description="Multi-Product Technical Support RAG API",
    version="1.0.0",
)


app.include_router(
    router,
    prefix="/api/v1",
)


@app.get("/health")
def health_check() -> dict:
    """Check whether the API is running."""

    return {
        "status": "healthy",
        "service": "TechSupportAI",
    }