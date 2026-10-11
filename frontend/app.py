
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1/query"

st.set_page_config(
    page_title="TechSupportAI",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ TechSupportAI")
st.subheader("Technical Hardware & Software Resolution Hub")
st.write(
    "Describe your technical issue and get troubleshooting guidance "
    "grounded in the knowledge base."
)

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
        with st.spinner("Searching the knowledge base..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                # Adjust these keys if your API returns a different schema.
                answer = (
                    data.get("answer")
                    or data.get("response")
                    or data.get("result")
                )

                if answer is None:
                    st.error(
                        "The API responded, but the answer field "
                        "was not found in its JSON response."
                    )
                    st.json(data)
                else:
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

            except requests.RequestException as error:
                st.error(
                    "Could not connect to TechSupportAI. "
                    "Check that the FastAPI backend is running."
                )
                st.caption(str(error))
