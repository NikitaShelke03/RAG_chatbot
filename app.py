from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# STEP 1: Load document
loader = TextLoader("sample.txt")
documents = loader.load()

# STEP 2: Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# STEP 3: Create embeddings
embeddings = HuggingFaceEmbeddings()

# STEP 4: Store in vector DB
db = FAISS.from_documents(docs, embeddings)

# STEP 5: Query system
query = input("Enter your question: ")

retrieved_docs = db.similarity_search(query)

# STEP 6: Use FLAN-T5 (FREE LLM)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# Create context
context = " ".join([doc.page_content for doc in retrieved_docs])

# Prompt (IMPORTANT FIX)
prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer in short:
"""

# Generate answer
result = qa_pipeline(
    prompt,
    max_new_tokens=80,
    do_sample=False
)

# Clean output (IMPORTANT FIX - removes repetition)
answer = result[0]["generated_text"].replace(prompt, "").strip()

print("\nAnswer:", answer)