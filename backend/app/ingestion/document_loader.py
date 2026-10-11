from pathlib import Path
from langchain_core.documents import Document
# Import the function you just fixed and verified!


def load_documents(directory: Path) -> list[Document]:
    """
    Load all TXT documents from the knowledge-base directory.

    Each file becomes one LangChain Document with:
    - page_content: the actual document text
    - metadata: source filename and file path
    """
    documents: list[Document] = []

    # Verify that the directory path actually exists to prevent crashes
    if not directory.exists():
        print(f"⚠️ Warning: Directory '{directory}' does not exist.")
        return []

    for file_path in sorted(directory.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                },
            )
        )

    print(f"📦 Successfully loaded {len(documents)} raw files from disk.")
    return documents