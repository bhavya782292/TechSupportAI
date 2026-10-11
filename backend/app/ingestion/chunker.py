from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.core.config import settings


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    child_documents = splitter.split_documents(documents)

    for index, doc in enumerate(child_documents):
        # Using "doc" instead of "document" avoids confusion
        doc.metadata["parent_id"] = doc.metadata.get("source", "unknown")
        doc.metadata["chunk_id"] = index

    return child_documents

if __name__ == "__main__":
    # Create a mock document to test with
    test_doc = Document(
        page_content="This is sentence one. This is sentence two. This is sentence three.",
        metadata={"source": "test_file.txt"}
    )

    # Run your function
    chunks = split_documents([test_doc])

    # Print out what happened
    print(f"Successfully created {len(chunks)} text chunks!")
    for chunk in chunks:
        print(f"-> Content: '{chunk.page_content}' | Metadata: {chunk.metadata}")
