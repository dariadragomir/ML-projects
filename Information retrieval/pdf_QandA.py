import PyPDF2
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import torch  
import json

def extract_text_from_pdf(pdf_path):
    pdf_document = PyPDF2.PdfReader(pdf_path)
    full_text = ""
    for page_num in range(len(pdf_document.pages)):
        page = pdf_document.pages[page_num] 
        full_text += page.extract_text() 
    return full_text

def chunk_text(text, chunk_size=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embeddings(texts, model_name='distilbert-base-nli-mean-tokens'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    
    index.add(embeddings.cpu().numpy())
    return index

def retrieve_relevant_chunks(query, chunks, index, model_name='distilbert-base-nli-mean-tokens'):
    query_embedding = generate_embeddings([query])[0]
    distances, indices = index.search(query_embedding.cpu().numpy().reshape(1, -1), k=5)  # top 5 results
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_answer(query, relevant_chunks, model):
    context = " ".join(relevant_chunks)
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained(model)
    
    inputs = tokenizer(f'question: {query} context: {context}', return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(inputs['input_ids'])
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    
    chunks = chunk_text(text)
    
    embeddings = generate_embeddings(chunks)
    
    index = create_faiss_index(embeddings)
    
    relevant_chunks = retrieve_relevant_chunks(question, chunks, index)
    
    answer = generate_answer(question, relevant_chunks, 't5-base')
    
    return answer

pdf_path = 'CV 2024.pdf'
question = 'What job does this suit for me?'
answer = main(pdf_path, question)
print(answer)
