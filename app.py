import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot")

# ================================
# ✅ STEP 1: Load Document
# ================================
@st.cache_data
def load_docs():
    loader = TextLoader("sample.txt")
    documents = loader.load()
    return documents

documents = load_docs()

# ================================
# ✅ STEP 2: Split Text
# ================================
@st.cache_data
def split_docs(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# ================================
# ✅ STEP 3: Embeddings
# ================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ================================
# ✅ STEP 4: Vector Store
# ================================
@st.cache_resource
def create_vectorstore(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

db = create_vectorstore(docs, embeddings)

# ================================
# ✅ STEP 5: Load FLAN-T5 Model
# ================================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# ================================
# ✅ STEP 6: User Input
# ================================
query = st.text_input("💬 Ask your question:")

if query:
    # Retrieve relevant chunks
    retrieved_docs = db.similarity_search(query, k=3)

    # Combine context
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Prompt
    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}

    Answer in a clear and concise way:
    """

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ================================
    # ✅ Output
    # ================================
    st.write("### ✅ Answer:")
    st.write(answer)

    # Optional: Show retrieved chunks
    with st.expander("📚 Retrieved Context"):
        for i, doc in enumerate(retrieved_docs):
            st.write(f"**Chunk {i+1}:**")
            st.write(doc.page_content)