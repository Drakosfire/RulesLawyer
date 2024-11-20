import json
import pandas as pd
from docling_core.types.doc.document import DoclingDocument, DocItemLabel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_document(json_path: str, chunk_size: int = 2000, overlap: int = 500):
    """Process document maintaining document structure and hierarchy."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    doc = DoclingDocument(**data)
    
    chunks_with_metadata = []
    current_headers = {}  # page_no -> current header text
    
    # Add document summary if available
    if 'document_summary' in data:
        chunks_with_metadata.append({
            'content': f"Document Summary:\n{data['document_summary']}",
            'page': 0,  # Use 0 for document-level content
            'content_type': 'summary',
            'document_name': doc.name if hasattr(doc, 'name') else '',
        })
    
    # Process document by page
    for page_no in doc.pages:
        page = doc.pages[page_no]
        current_header = None
        
        # Add page summary if available
        if hasattr(page, 'summary'):
            chunks_with_metadata.append({
                'content': f"Page {page_no} Summary:\n{page.summary}",
                'page': page_no,
                'content_type': 'page_summary',
                'document_name': doc.name if hasattr(doc, 'name') else '',
            })
        
        # Process page content
        page_items = list(doc.iterate_items(page_no=page_no))
        
        for item, _ in page_items:
            # Create base metadata
            metadata = {
                'page': page_no,
                'current_section': current_header,
                'content_type': item.label.value,
                'document_name': doc.name if hasattr(doc, 'name') else '',
            }
            
            # Track section headers
            if item.label == DocItemLabel.SECTION_HEADER:
                current_header = item.text
                current_headers[page_no] = item.text
                
                # Convert header to markdown
                md_content = f"# {item.text}\n"
            
            # Handle regular text
            elif item.label == DocItemLabel.TEXT:
                # Convert text to markdown with context
                md_content = ""
                if current_header:
                    md_content += f"Context: {current_header}\n\n"
                md_content += f"{item.text}\n"
            
            else:
                # Skip page headers/footers and other non-content elements
                continue
            
            # Add provenance data if available
            if hasattr(item, 'prov') and item.prov:
                metadata['bbox'] = item.prov[0].bbox.as_tuple()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            chunks = text_splitter.split_text(md_content)
            
            # Store chunks with metadata
            for chunk in chunks:
                chunk_metadata = {
                    'content': chunk,
                    **metadata,
                    'original_text': item.orig,
                }
                chunks_with_metadata.append(chunk_metadata)
    
    return create_dataframe(chunks_with_metadata)

def create_dataframe(chunks_with_metadata):
    """Create DataFrame with content and available metadata."""
    # Add index to chunks
    for i, chunk in enumerate(chunks_with_metadata):
        chunk['chunk_index'] = i
    
    # Get content in specific order
    contents = [c['content'] for c in chunks_with_metadata]
    
    # Create embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    embeddings = embeddings_model.embed_documents(contents)
    
    # Create DataFrame with index verification and safe access to optional fields
    df = pd.DataFrame({
        'chunk_index': [c['chunk_index'] for c in chunks_with_metadata],
        'content': contents,
        'embedding': [json.dumps(e) for e in embeddings],
        'page': [c.get('page', None) for c in chunks_with_metadata],
        'section': [c.get('current_section', '') for c in chunks_with_metadata],
        'content_type': [c.get('content_type', '') for c in chunks_with_metadata],
        'original_text': [c.get('original_text', '') for c in chunks_with_metadata],
        'bbox': [c.get('bbox', None) for c in chunks_with_metadata],
    })
    
    # Verify alignment
    assert all(df['chunk_index'] == range(len(df))), "Chunk order mismatch!"
    
    return df