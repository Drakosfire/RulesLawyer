import json
import sys
import os
from document_processing.utilities import get_file_name_without_ext
from docling_core.types.doc.document import DoclingDocument, DocItemLabel
# Get the correct path to docling-core
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# docling-core is in the same directory as the script
docling_core_path = os.path.join(current_dir, 'docling-core')


# Add to Python path
sys.path.insert(0, docling_core_path)

# Now try the import
from docling_core.types.doc.document import DoclingDocument

def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_text_by_page(data):
    """Extract text while preserving original page structure."""
    pages = {}
    
    # Initialize pages structure
    for page_no, page_data in data.get('pages', {}).items():
        pages[page_no] = {
            'size': page_data['size'],
            'page_no': page_data['page_no'],
            'text_entries': {},
            'image': page_data.get('image', None)
        }
    
    # Directly process texts array
    for text_item in data.get('texts', []):
        if 'text' in text_item and 'prov' in text_item and text_item['prov']:
            page_number = str(text_item['prov'][0]['page_no'])
            if page_number in pages:
                entry_number = len(pages[page_number]['text_entries']) + 1
                pages[page_number]['text_entries'][f"entry_{entry_number}"] = {
                    'text': text_item['text'],
                    'label': text_item.get('label', ''),
                    'level': text_item.get('level', None)
                }
    
    return pages

def save_enhanced_json(data, pages, output_file):
    """Save enhanced JSON while preserving original structure."""
    # Create a deep copy to avoid modifying the original
    enhanced_data = json.loads(json.dumps(data))
    
    # Add text entries and summaries to pages
    for page_no, page_data in pages.items():
        if str(page_no) in enhanced_data['pages']:
            enhanced_data['pages'][str(page_no)]['text_entries'] = page_data['text_entries']
            if 'summary' in page_data:
                enhanced_data['pages'][str(page_no)]['summary'] = page_data['summary']
    
    # Add document summary
    if 'document_summary' in data:
        enhanced_data['document_summary'] = data['document_summary']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

def collect_headers_from_json(json_path):
    """Collect headers organized by page from the JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    doc = DoclingDocument(**data)
    
    headers_by_page = {}
    for page_no in doc.pages:
        page_headers = []
        for item, _ in doc.iterate_items(page_no=page_no):
            if item.label == DocItemLabel.SECTION_HEADER:
                page_headers.append(item.text)
        headers_by_page[page_no] = page_headers
    
    return headers_by_page
def page_enhanced_markdown_export(file_path):
    file_name_without_ext = get_file_name_without_ext(file_path)
    # Load the parsed PDF data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Print the structure of the loaded data
    print("Data structure:")
    print(json.dumps(data, indent=2)[:500])  # Print first 500 characters of formatted JSON

    # Create a DoclingDocument instance directly from the loaded data
    try:
        doc = DoclingDocument(**data)
    except Exception as e:
        print(f"Error creating DoclingDocument: {e}")
        return

    # Export to markdown
    try:
        markdown_output = doc.export_to_markdown(include_page_numbers=True)
    except Exception as e:
        print(f"Error exporting to markdown: {e}")
        return

    # Print the first 1000 characters of the output
    print("\nMarkdown output (first 1000 characters):")
    print(markdown_output[:1000])

    # Optionally, save the full output to a file
    with open(f'./output/{file_name_without_ext}.md', 'w') as f:
        f.write(markdown_output)
    