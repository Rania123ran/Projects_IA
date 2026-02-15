from pypdf import PdfReader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
def process_text(txt): 
    text_sp = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_sp.split_text(txt)
    embedding = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
    db_vect = FAISS.from_texts(chunks,embedding)
    return db_vect
def summarizer(pdf):
    if pdf is not None : 
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages : 
            page_text = page.extract_text()
            if page_text : 
                text += page_text + " \n" 
        db_vect = process_text(text)
        query = "Summarize the content of the uploaded PDF File in approximately 3-5 sentences"
        if query : 
            docs = db_vect.similarity_search(query)
            llm = Ollama(model="phi3",temperature=0.1)
            chain = load_qa_chain(llm,chain_type="stuff")
            resp = chain.run(input_documents=docs,question=query)
            return resp



