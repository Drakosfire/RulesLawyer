import json

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