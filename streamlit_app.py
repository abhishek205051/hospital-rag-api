import streamlit as st
import requests

st.set_page_config(
    page_title="Hospital RAG Assistant",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Hospital RAG Assistant")
st.write("Ask questions about the hospital documents.")

API_URL = "http://localhost:8000/query"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"query": question},
                    timeout=60,
                )

                response.raise_for_status()
                result = response.json()

                answer = result.get("answer", "No answer returned.")

                st.markdown(answer)

                # Optional: show retrieved documents
                sources = result.get("sources", [])

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(source)

            except requests.RequestException as e:
                answer = f"API error: {e}"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })