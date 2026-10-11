import json

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.ingestion.chunker import split_documents
from backend.app.ingestion.document_loader import load_documents
from backend.app.retrieval.bm25_store import bm25_store
from backend.app.retrieval.retrieval_pipeline import retrieval_pipeline


DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "out_of_scope_questions.json"
)


def evaluate_out_of_scope() -> None:
    """Inspect retrieval behavior for out-of-scope questions."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    parent_documents = load_documents(
        PROJECT_ROOT / "data" / "knowledge_base"
    )

    child_documents = split_documents(parent_documents)

    bm25_store.build_index(child_documents)

    for test_case in test_cases:
        question = test_case["question"]

        results = retrieval_pipeline.retrieve(
            query=question,
            parent_documents=parent_documents,
            candidate_k=settings.dense_top_k,
            final_k=settings.reranker_top_k,
        )

        print(f"\nQuestion: {question}")

        if not results:
            print("Result: No relevant documents")
            continue

        print("Retrieved:")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{result['child_id']} | "
                f"Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )


if __name__ == "__main__":
    evaluate_out_of_scope()