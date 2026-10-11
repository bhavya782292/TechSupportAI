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
    / "test_questions.json"
)


def evaluate_retrieval() -> None:
    """Evaluate retrieval using Hit@1, MRR, and reranker scores."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    parent_documents = load_documents(
        PROJECT_ROOT / "data" / "knowledge_base"
    )

    child_documents = split_documents(parent_documents)

    bm25_store.build_index(child_documents)

    hit_at_1 = 0
    reciprocal_ranks = []
    top_scores = []

    for test_case in test_cases:
        question = test_case["question"]
        expected_chunks = set(test_case["expected_chunks"])

        results = retrieval_pipeline.retrieve(
            query=question,
            parent_documents=parent_documents,
            candidate_k=settings.dense_top_k,
            final_k=settings.reranker_top_k,
        )

        retrieved_chunks = [
            result["child_id"]
            for result in results
        ]

        # Get the highest reranker score.
        top_score = (
            results[0]["reranker_score"]
            if results
            else None
        )

        if top_score is not None:
            top_scores.append(top_score)

        # Hit@1
        if retrieved_chunks:
            if retrieved_chunks[0] in expected_chunks:
                hit_at_1 += 1

        # Reciprocal Rank
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(
            retrieved_chunks,
            start=1,
        ):
            if chunk_id in expected_chunks:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {question}")
        print(f"Expected chunk: {sorted(expected_chunks)}")
        print(f"Retrieved chunks: {retrieved_chunks}")
        print(
            f"Top Reranker Score: "
            f"{top_score:.4f}"
            if top_score is not None
            else "Top Reranker Score: None"
        )
        print(f"Reciprocal Rank: {reciprocal_rank:.2f}")

    hit_at_1_score = (
        hit_at_1 / len(test_cases) * 100
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    print("\n" + "=" * 50)
    print(f"Hit@1: {hit_at_1_score:.2f}%")
    print(f"MRR: {mrr:.4f}")

    if top_scores:
        print("\nTop Reranker Scores:")
        for score in top_scores:
            print(f"{score:.4f}")

        print(
            f"\nMinimum in-scope score: "
            f"{min(top_scores):.4f}"
        )

        print(
            f"Maximum in-scope score: "
            f"{max(top_scores):.4f}"
        )

    print("=" * 50)


if __name__ == "__main__":
    evaluate_retrieval()