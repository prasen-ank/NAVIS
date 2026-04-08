import streamlit as st
import requests
import os
from pathlib import Path

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG File Chatbot", layout="wide")
st.title("RAG File Chatbot (Streamlit)")

with st.sidebar:
    st.header("Upload data files")
    namespace = st.text_input("Namespace (project)", value="default")
    uploaded = st.file_uploader("Choose a file", accept_multiple_files=True)
    if st.button("Upload files"):
        for f in uploaded:
            files = {"file": (f.name, f.getvalue())}
            resp = requests.post(f"{API_URL}/upload", files=files, data={"namespace": namespace})
            st.write(f"Uploaded {f.name}: {resp.status_code}")

    if st.button("Clear embedded data"):
        resp = requests.post(f"{API_URL}/clear", data={"namespace": namespace})
        if resp.status_code == 200:
            st.success(f"Cleared embedded data for namespace '{namespace}'")
            st.session_state.history = []
        else:
            st.error(f"Failed to clear namespace: {resp.status_code} {resp.text}")

st.header("Chat")
if "history" not in st.session_state:
    st.session_state.history = []

with st.form("chat_form"):
    question = st.text_input("Ask a question about your uploaded files")
    submitted = st.form_submit_button("Send")

if submitted and question:
    st.session_state.history.append({"user": question})
    data = {"namespace": namespace, "question": question}
    resp = requests.post(f"{API_URL}/chat", data=data)
    if resp.status_code == 200:
        j = resp.json()
        if j.get("type") == "image":
            url = f"{API_URL}{j.get('url')}"
            st.session_state.history.append({"bot_image": url})
        else:
            st.session_state.history.append({"bot": j.get("answer")})
    else:
        st.session_state.history.append({"bot": f"(error) {resp.status_code} {resp.text}"})

for item in st.session_state.history:
    if "user" in item:
        st.markdown(f"**You:** {item['user']}")
    if "bot" in item:
        st.markdown(f"**Bot:** {item['bot']}")
    if "bot_image" in item:
        st.image(item['bot_image'])
