import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Page setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Title
st.title("🤖 RAG Chatbot")

# -------------------------
# LOAD DATA (same as your code)
# -------------------------
@st.cache_resource
def load_db():
    loader = TextLoader("sample.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    return db

db = load_db()

# -------------------------
# LOAD MODEL (same as your code)
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = load_model()

# -------------------------
# CHAT MEMORY
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# DISPLAY CHAT (ChatGPT style)
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask your question...")

if query:
    # Show user message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # -------------------------
    # YOUR RAG LOGIC (unchanged)
    # -------------------------
    retrieved_docs = db.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {query}

    Answer in short:
    """

    result = qa_pipeline(
        prompt,
        max_new_tokens=80,
        do_sample=False
    )

    answer = result[0]["generated_text"].replace(prompt, "").strip()

    # -------------------------
    # SHOW BOT RESPONSE
    # -------------------------
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})