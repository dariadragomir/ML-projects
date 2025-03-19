import os
import sys
import argparse
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

def create_index(directory):
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    
    schema = Schema(title=ID(stored=True), content=TEXT(stored=True))
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                writer.add_document(title=filename, content=content)
    
    writer.commit()
    print("Indexing completed.")

def search_index(query_str):
    ix = open_dir("indexdir")
    
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query, limit=5)
        for result in results:
            print(result["title"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-index", action="store_true", help="Index documents")
    parser.add_argument("-search", action="store_true", help="Search documents")
    parser.add_argument("-directory", type=str, help="Directory of documents to index")
    parser.add_argument("-query", type=str, help="Query string to search")
    
    args = parser.parse_args()
    
    if args.index and args.directory:
        create_index(args.directory)
    elif args.search and args.query:
        search_index(args.query)
    else:
        print("Invalid command. Use -index with -directory or -search with -query.")

if __name__ == "__main__":
    main()
