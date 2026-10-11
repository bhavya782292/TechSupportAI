
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------- APP CONFIG ----------------

st.set_page_config(
    page_title="TechSupportAI",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ TechSupportAI")
st.caption("Technical Hardware & Software Resolution Hub")

# ---------------- API KEY ----------------

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("Add GOOGLE_API_KEY in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ---------------- KNOWLEDGE BASE ----------------

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
    st.error("No .txt files found in data/knowledge_base.")
    st.stop()


@st.cache_resource
def build_search_index(texts):
    tokenized = [text.lower().split() for text in texts]
    return BM25Okapi(tokenized)


search_index = build_search_index(
    tuple(item["text"] for item in chunks)
)

# ---------------- GEMINI MODEL ----------------

@st.cache_resource
def get_model(key):
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=key,
        temperature=0,
        max_output_tokens=300,
    )


model = get_model(api_key)

# ---------------- RETRIEVAL ----------------

def retrieve_context(question):
    scores = search_index.get_scores(question.lower().split())

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    selected = [
        i for i in ranked if scores[i] > 0
    ][:2]

    if not selected:
        return ""

    return "\n\n".join(
        f"Source: {chunks[i]['source']}\n"
        f"{chunks[i]['text'][:800]}"
        for i in selected
    )


# ---------------- RESPONSE CLEANUP ----------------

def clean_response(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(parts).strip()

    return str(content)


# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT ----------------

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
            with st.spinner("Finding the best troubleshooting steps..."):
                context = retrieve_context(question)

                if not context:
                    answer = (
                        "I couldn't find a relevant answer in the "
                        "knowledge base. Please provide the device "
                        "model and describe the symptoms."
                    )
                else:
                    system_prompt = """
You are TechSupportAI, a concise technical support assistant.

Use only the provided knowledge-base context.
Identify the user's specific device and issue.
Give up to 3 relevant, numbered troubleshooting steps.
Use short, clear sentences.
Do not invent product-specific facts or unrelated advice.
If the context is insufficient, say so.
"""

                    response = model.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=(
                            f"Knowledge-base context:\n{context}\n\n"
                            f"User issue: {question}"
                        )),
                    ])

                    answer = clean_response(response.content)

                    if not answer:
                        answer = "No readable response was returned. Please retry."

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

                except Exception as error:
            error_text = str(error)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                st.warning(
                    "The AI service has reached its free usage limit. "
                    "Please try again after the quota resets."
                )
            else:
                st.error("Couldn't generate an answer. Please try again.")
                st.caption(error_text)
