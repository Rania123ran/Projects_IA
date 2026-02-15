import streamlit as st
import pdfplumber
import io

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

st.title("Assistant Révision Deep Learning")

@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(model_name="phi3"):
    return ChatOllama(model=model_name, temperature=0)

@st.cache_resource
def build_faiss_index(pdf_bytes: bytes):
    documents = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": i}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, get_embedder())
    return db

uploaded_file = st.file_uploader("Upload un PDF", type="pdf")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    db = build_faiss_index(pdf_bytes)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    question = st.text_input("Pose ta question")

    ask = st.button("Répondre")

    if ask and question:
        docs = retriever.invoke(question)

        context = "\n\n".join(
            [f"(Page {d.metadata['page']}) {d.page_content}" for d in docs]
        )

        prompt = f"""
Réponds en français.
Utilise uniquement le contexte.
Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".

Contexte:
{context}

Question: {question}
Réponse:
"""

        llm = get_llm("phi3")  # ou "llama3" si tu veux plus puissant
        response = llm.invoke(prompt)

        st.subheader("Réponse :")
        st.write(response.content)
