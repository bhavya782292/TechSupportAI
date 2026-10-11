
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------- APP CONFIG --------------------

st.set_page_config(
    page_title="TechSupportAI",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ TechSupportAI")
st.caption("Technical Hardware & Software Resolution Hub")

# -------------------- API CONFIG --------------------

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error(
        "Missing GOOGLE_API_KEY. Add it in "
        "Streamlit Cloud → Manage app → Settings → Secrets."
    )
    st.stop()

# -------------------- KNOWLEDGE BASE --------------------

KB_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "knowledge_base"
)

@st.cache_data
def load_chunks():
    documents = []

    if not KB_DIR.exists():
        return documents

    for file_path in sorted(KB_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        # Keep chunks small to reduce processing time.
        for start in range(0, len(text), 900):
            chunk = text[start:start + 1100].strip()

            if chunk:
                documents.append({
                    "source": file_path.name,
                    "text": chunk,
                })

    return documents


chunks = load_chunks()

if not chunks:
    st.error(
        "No .txt knowledge-base files found. "
        "Check data/knowledge_base in your GitHub repository."
    )
    st.stop()


@st.cache_resource
def build_search_index(texts):
    tokenized_documents = [
        text.lower().split() for text in texts
    ]
    return BM25Okapi(tokenized_documents)


search_index = build_search_index(
    tuple(item["text"] for item in chunks)
)

# -------------------- GEMINI MODEL --------------------

@st.cache_resource
def get_model(key):
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=key,
        temperature=0,
    )


model = get_model(api_key)

# -------------------- CHAT HISTORY --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- RETRIEVAL --------------------

def retrieve_context(question):
    scores = search_index.get_scores(question.lower().split())

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected = [
        index for index in ranked_indices
        if scores[index] > 0
    ][:3]

    if not selected:
        return ""

    return "\n\n---\n\n".join(
        f"Source: {chunks[index]['source']}\n"
        f"{chunks[index]['text'][:900]}"
        for index in selected
    )


# -------------------- RESPONSE CLEANUP --------------------

def clean_response(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content)


# -------------------- ANSWER GENERATION --------------------

question = st.chat_input("Describe your technical issue...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the knowledge base..."):
                context = retrieve_context(question)

                if not context:
                    answer = (
                        "I couldn't find relevant information in the "
                        "knowledge base. Please mention the device, "
                        "model, and symptoms for more specific guidance."
                    )
                else:
                    system_prompt = """
You are TechSupportAI, a technical support assistant.

Rules:
1. Identify the device and the exact issue described by the user.
2. Use the supplied knowledge-base context to answer.
3. Give clear, numbered troubleshooting steps.
4. Keep the answer concise and relevant to the reported problem.
5. Do not claim you tested or repaired the device.
6. Never invent model-specific instructions or facts.
7. If the context is insufficient, explain what information is missing.
8. Include safety precautions when relevant.
9. Do not return raw JSON, Python objects, or internal metadata.
10. If the user reports a router issue, distinguish between no internet,
    Wi-Fi disconnections, power problems, and slow connectivity when possible.
"""

                    response = model.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=(
                            f"Knowledge-base context:\n{context}\n\n"
                            f"User's issue: {question}\n\n"
                            "Provide the most relevant troubleshooting steps."
                        )),
                    ])

                    answer = clean_response(response.content)

                    if not answer:
                        answer = (
                            "I couldn't generate a readable response. "
                            "Please try again."
                        )

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as error:
            st.error(
                "Something went wrong while generating the answer. "
                "Please try again."
            )
            st.caption(f"Technical details: {error}")
