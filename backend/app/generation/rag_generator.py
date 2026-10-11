
import time

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.generation.llm_service import llm_service
from backend.app.generation.prompt import RAG_PROMPT
from backend.app.ingestion.chunker import split_documents
from backend.app.ingestion.document_loader import load_documents
from backend.app.retrieval.bm25_store import bm25_store
from backend.app.retrieval.retrieval_pipeline import retrieval_pipeline


class RAGGenerator:
    """Generate grounded answers using retrieved knowledge-base context."""

    def generate(self, question: str) -> str:
        print("[DEBUG] RAG generation started", flush=True)
        total_start = time.perf_counter()

        # 1. Load parent documents.
        start = time.perf_counter()
        parent_documents = load_documents(
            PROJECT_ROOT / "data" / "knowledge_base"
        )
        print(
            f"[TIMING] Document loading: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # 2. Split documents and build BM25 index.
        start = time.perf_counter()
        child_documents = split_documents(parent_documents)
        bm25_store.build_index(child_documents)
        print(
            f"[TIMING] Chunking + BM25 indexing: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # 3. Retrieve and rerank relevant context.
        start = time.perf_counter()
        retrieved_results = retrieval_pipeline.retrieve(
            query=question,
            parent_documents=parent_documents,
            candidate_k=settings.dense_top_k,
            final_k=settings.reranker_top_k,
        )
        print(
            f"[TIMING] Retrieval pipeline: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if not retrieved_results:
            print("[TIMING] No relevant documents found.", flush=True)
            return "I could not find relevant information in the knowledge base."

        # 4. Build context and prompt.
        start = time.perf_counter()
        context_parts = [
            result["parent_document"]
            for result in retrieved_results
        ]
        context = "\n\n---\n\n".join(context_parts)

        prompt_value = RAG_PROMPT.invoke(
            {
                "context": context,
                "question": question,
            }
        )
        print(
            f"[TIMING] Context + prompt preparation: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        # 5. Generate the answer using Gemini.
        start = time.perf_counter()
        answer = llm_service.generate(prompt_value)
        print(
            f"[TIMING] Gemini generation: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        print(
            f"[TIMING] TOTAL request processing: "
            f"{time.perf_counter() - total_start:.2f}s",
            flush=True,
        )

        return answer


rag_generator = RAGGenerator()
