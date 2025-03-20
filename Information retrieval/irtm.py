import os
import argparse
import PyPDF2
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

def extract_text_from_pdf(pdf_path):
    pdf_document = PyPDF2.PdfReader(pdf_path)
    full_text = ""
    for page_num in range(len(pdf_document.pages)):
        page = pdf_document.pages[page_num]
        full_text += page.extract_text() + "\n"
    return full_text

def create_index(pdf_path):
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    
    schema = Schema(title=ID(stored=True), content=TEXT(stored=True))
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    
    text = extract_text_from_pdf(pdf_path)
    writer.add_document(title=os.path.basename(pdf_path), content=text)
    writer.commit()
    print("Indexing completed.")

def search_index(query_str):
    ix = open_dir("indexdir")
    
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query, limit=5)
        for result in results:
            print("Relevant Section from CV:")
            print(result["content"]) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-index", action="store_true", help="Index the CV PDF")
    parser.add_argument("-search", type=str, help="Search query to find relevant information in CV")
    parser.add_argument("-file", type=str, help="Path to the CV PDF file")
    
    args = parser.parse_args()
    
    if args.index and args.file:
        create_index(args.file)
    elif args.search:
        search_index(args.search)
    else:
        print("Invalid command. Use -index with -file or -search with a query.")

if __name__ == "__main__":
    main()
