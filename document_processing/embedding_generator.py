import json
import pandas as pd
from typing import List, Dict, Any
import re
import time
from docling_core.types.doc.document import DoclingDocument, DocItemLabel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_processing.json_utils import collect_headers_from_json

class SimplePageAwareSplitter(RecursiveCharacterTextSplitter):
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        # First, let's split the text into pages
        pages = re.split(r'\[PAGE(\d+)\]', text)
        print(f"pages: {pages}")
        result = []
        for i in range(0, len(pages), 2):
            # print(f"i: {i}")
            if i == 0:
                content = pages[i]
                page_number = 1
            else:
                page_number = int(pages[i-1])
                content = pages[i]
                print(f"content: {content}")
            
            chunks = super().split_text(content)
            
            # Add page number to each chunk
            result.extend([{'content': chunk, 'page': page_number} for chunk in chunks])
        
        return result

# Function to read the Markdown file
def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def collect_document_summary_from_json(json_data):
    return json_data.get('document_summary', '')

def collect_page_summary_from_json(json_data):
    """Collects summaries for each page from the JSON data."""
    page_summaries = {}
    pages = json_data.get('pages', {})
    for page_no, page_data in pages.items():
        page_summaries[page_no] = page_data.get('summary', "")
    print(f"page_summaries: {page_summaries}")
    return page_summaries

def process_document(md_path: str, json_path: str, chunk_size: int = 2000, overlap: int = 500):
    """Process document maintaining document structure and hierarchy."""

    # Load the model
    print("Loading the embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    print("Embedding model loaded successfully.")

    md_data = read_markdown(md_path)
    json_data = read_json(json_path)

    # Collect headers by page
    headers_by_page = collect_headers_from_json(json_path)
    
# Create text splitter
    print("Chunking the text...")
    text_splitter = SimplePageAwareSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

     # Split the text    
    chunks = text_splitter.split_text(md_data)
    embeddings = embeddings_model.embed_documents([chunk['content'] for chunk in chunks])
    headers_by_page = collect_headers_from_json(json_path)

    # Collect page summaries
    page_summaries = collect_page_summary_from_json(json_data)

    # Add headers and summaries to chunks
    print(f"Number of chunks: {len(chunks)}")
    for i in range(len(chunks)):
        print(f"Chunk {i}: ")
        page_num = chunks[i]['page']
        print(f"Page number: {page_num}")
        
        # Ensure the page number exists in headers_by_page
        if page_num in headers_by_page:
            chunks[i]['sections'] = headers_by_page[page_num]
        else:
            chunks[i]['sections'] = []
            print(f"Warning: Page {page_num} not found in headers_by_page")
        
        # Add page summary to each chunk
        if str(page_num) in page_summaries:
            chunks[i]['page_summary'] = page_summaries[str(page_num)]
        else:
            chunks[i]['page_summary'] = ""
            print(f"Warning: Page {page_num} not found in page_summaries")

    # Add document summary to the first chunk
    chunks[0]['document_summary'] = collect_document_summary_from_json(json_data)
    

    # Create a DataFrame with chunks and embeddings
    print("\nCreating DataFrame...")
    df = pd.DataFrame({
        'page': [chunk['page'] for chunk in chunks],
        'sections': [json.dumps(chunk['sections']) for chunk in chunks], # Headers from the page
        'content': [chunk['content'] for chunk in chunks],
        'chunk_char_count': [len(chunk['content']) for chunk in chunks],
        'chunk_word_count': [len(chunk['content'].split()) for chunk in chunks],
        'chunk_token_count': [len(chunk['content'].split()) * 1.3 for chunk in chunks],
        'embedding': [json.dumps(emb) for emb in embeddings],
    })

    # Remove rows with sentence_chunks less than 200 characters
    # print("Removing rows with sentence chunks less than 200 characters...")
    df = df[df['chunk_char_count'] >= 200]

    return df