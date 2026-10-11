
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="TechSupportAI", page_icon="🛠️")
st.title("🛠️ TechSupportAI")
st.caption("Technical Hardware & Software Resolution Hub")

# Set GOOGLE_API_KEY in Streamlit Secrets.
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("Add GOOGLE_API_KEY in Streamlit → Settings → Secrets.")
    st.stop()

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"

@st.cache_data
def load_chunks():
    chunks = []
    for file in sorted(KB_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        # Lightweight chunks; no embedding or reranker models.
        for start in range(0, len(text), 900):
            chunk = text[start:start + 1100].strip()
            if chunk:
                chunks.append((file.name, chunk))
    return chunks

chunks = load_chunks()

if not chunks:
    st.error(f"No .txt knowledge-base files found in: {KB_DIR}")
    st.stop()

@st.cache_resource
def get_model(key):
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=key,
        temperature=0,
    )
model = get_model(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Describe your technical issue...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            tokenized = [text.lower().split() for _, text in chunks]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(question.lower().split())
            best = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:4]

            context = "\n\n".join(
                f"Source: {chunks[i][0]}\n{chunks[i][1]}"
                for i in best if scores[i] > 0
            )

            if not context:
                answer = (
                    "I couldn't find relevant information in the "
                    "knowledge base for that question."
                )
            else:
                response = model.invoke([
                    SystemMessage(content=(
                        "You are TechSupportAI. Answer using only the "
                        "provided knowledge-base context. Do not invent "
                        "facts or troubleshooting steps. If the context "
                        "is insufficient, say so. Be concise and practical."
                    )),
                    HumanMessage(content=(
                        f"Knowledge-base context:\n{context}\n\n"
                        f"Question: {question}"
                    )),
                ])
                answer = response.content

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as error:
            st.error(f"Error: {error}")
