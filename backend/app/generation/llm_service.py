
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from backend.app.core.config import settings


class LLMService:
    """Manage communication with the Gemini language model."""

    def __init__(self) -> None:
        self.model = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
            timeout=45,
            max_retries=1,
        )

    @traceable(name="TechSupportAI_Gemini_Generation")
    def generate(self, prompt) -> str:
        """Send a prompt to Gemini and return clean generated text."""

        start = time.perf_counter()
        print("[DEBUG] Sending request to Gemini...", flush=True)

        try:
            response = self.model.invoke(prompt)

            elapsed = time.perf_counter() - start
            print(
                f"[DEBUG] Gemini responded in {elapsed:.2f}s",
                flush=True,
            )

            if isinstance(response.content, str):
                return response.content.strip()

            if isinstance(response.content, list):
                text_parts = []

                for item in response.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))

                return "".join(text_parts).strip()

            return str(response.content)

        except Exception as error:
            elapsed = time.perf_counter() - start
            print(
                f"[ERROR] Gemini failed after {elapsed:.2f}s: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            raise


llm_service = LLMService()
