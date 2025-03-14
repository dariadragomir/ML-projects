import os
import PyPDF2
import torch
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def chunk_text(text, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return [embeddings_model.embed_query(chunk) for chunk in chunks]


def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

def query_rag(index, query_text, chunks, top_k=3):
    query_embedding = np.array([embeddings_model.embed_query(query_text)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [chunks[i] for i in indices[0]]
    return "\n".join(retrieved_docs)


def answer_question(llm, retriever, query):
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain.run(query)

def main(pdf_path, query_text):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    retrieved_text = query_rag(index, query_text, chunks)
    print("Relevant Retrieved Text:\n", retrieved_text)
    
