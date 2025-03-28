import os
import re
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PDFRAGSystem:
    def __init__(self, pdf_path=None, qa_model_name="deepset/xlm-roberta-large-squad2"):
        self.pdf_path = pdf_path
        self.qa_model_name = qa_model_name
        self.context_chunks = []
        self.embeddings = None
        
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = 0 
        self.device = device
        
        self.qa_model = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name, device=device)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if pdf_path:
            self.load_and_process_pdf(pdf_path)
    
    def load_and_process_pdf(self, pdf_path):
        self.pdf_path = pdf_path
        raw_chunks = self._extract_text_from_pdf(pdf_path)
        self.context_chunks = self._refine_chunks(raw_chunks)
        self.embeddings = self._generate_embeddings(self.context_chunks)
    
    def _extract_text_from_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        text_chunks = []
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    text_chunks.extend(paragraphs)
        return text_chunks
    
    def _refine_chunks(self, chunks, max_chunk_words=150):
        refined_chunks = []
        for chunk in chunks:
            words = chunk.split()
            if len(words) > max_chunk_words:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                temp_chunk = ""
                for sentence in sentences:
                    if len((temp_chunk + " " + sentence).split()) > max_chunk_words:
                        if temp_chunk:
                            refined_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        temp_chunk += " " + sentence
                if temp_chunk:
                    refined_chunks.append(temp_chunk.strip())
            else:
                refined_chunks.append(chunk)
        return refined_chunks
    
    def _generate_embeddings(self, text_chunks):
        return self.embedding_model.encode(text_chunks, convert_to_tensor=True)
    
    def _find_relevant_context(self, question, top_k=5):
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        similarities = cosine_similarity(
            question_embedding.reshape(1, -1).cpu(),
            self.embeddings.cpu().reshape(len(self.context_chunks), -1)
        )
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        return [self.context_chunks[i] for i in top_indices]
    
    def ask_question(self, question, top_k_context=5):
        if not self.context_chunks:
            raise ValueError("No PDF loaded. Please load a PDF first.")
        
        relevant_contexts = self._find_relevant_context(question, top_k=top_k_context)
        combined_context = " ".join(relevant_contexts)
        
        result = self.qa_model(question=question, context=combined_context)
        
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'context_used': relevant_contexts
        }

if __name__ == "__main__":
    pdf_path = "CV_Daria Dragomir.pdf" 
    rag_system = PDFRAGSystem(pdf_path)
    
    #question = "Care este numele complet menționat în CV?"
    #question = "Ce abilități tehnice ai?"
    #question = "Unde ai studiat?"
    #question = "Care este numărul de telefon?"
    question = "Ce adresa de e-mail am?"
    answer = rag_system.ask_question(question)
    
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']:.2f}")
    print("\nContext used:")
    for i, ctx in enumerate(answer['context_used'], 1):
        print(f"{i}. {ctx[:200]}...") 
