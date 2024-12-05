import pytest
import json
import os
from document_processing.json_utils import (
    save_to_file,
    load_json,
    extract_text_by_page,
    save_enhanced_json,
    extract_headers_by_page
)

# Fixtures
@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / "test_file.txt"

@pytest.fixture
def temp_json_file(tmp_path):
    return tmp_path / "test_file.json"

@pytest.fixture
def sample_data():
    return {
        "pages": {
            "1": {
                "size": {"width": 612, "height": 792},
                "page_no": 1,
                "image": "base64_encoded_string"
            },
            "2": {
                "size": {"width": 612, "height": 792},
                "page_no": 2,
                "image": None
            }
        },
        "texts": [
            {
                "text": "Chapter 1",
                "label": "section_header",
                "level": 1,
                "prov": [{"page_no": 1}]
            },
            {
                "text": "Regular text",
                "label": "",
                "level": None,
                "prov": [{"page_no": 1}]
            },
            {
                "text": "Chapter 2",
                "label": "section_header",
                "level": 1,
                "prov": [{"page_no": 2}]
            }
        ]
    }

def test_save_to_file(temp_file):
    content = "Test content"
    save_to_file(content, temp_file)
    
    assert os.path.exists(temp_file)
    with open(temp_file, 'r', encoding='utf-8') as f:
        assert f.read() == content

def test_load_json(temp_json_file):
    test_data = {"key": "value"}
    with open(temp_json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    loaded_data = load_json(temp_json_file)
    assert loaded_data == test_data

def test_extract_text_by_page(sample_data):
    pages = extract_text_by_page(sample_data)
    
    assert "1" in pages
    assert "2" in pages
    
    # Check page 1
    assert pages["1"]["size"] == {"width": 612, "height": 792}
    assert pages["1"]["page_no"] == 1
    assert len(pages["1"]["text_entries"]) == 2
    assert pages["1"]["text_entries"]["entry_1"]["text"] == "Chapter 1"
    assert pages["1"]["text_entries"]["entry_1"]["label"] == "section_header"
    
    # Check page 2
    assert len(pages["2"]["text_entries"]) == 1
    assert pages["2"]["text_entries"]["entry_1"]["text"] == "Chapter 2"

def test_save_enhanced_json(temp_json_file, sample_data):
    pages = extract_text_by_page(sample_data)
    # Add a summary to test that feature
    pages["1"]["summary"] = "Page 1 summary"
    sample_data["document_summary"] = "Document summary"
    
    save_enhanced_json(sample_data, pages, temp_json_file)
    
    with open(temp_json_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    assert "document_summary" in saved_data
    assert saved_data["pages"]["1"]["summary"] == "Page 1 summary"
    assert len(saved_data["pages"]["1"]["text_entries"]) == 2

def test_extract_headers_by_page(sample_data):
    pages = extract_text_by_page(sample_data)
    headers = extract_headers_by_page({"pages": {
        "1": pages["1"],
        "2": pages["2"]
    }})
    
    assert headers["1"] == ["Chapter 1"]
    assert headers["2"] == ["Chapter 2"]

# Error cases
def test_load_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_json("nonexistent_file.json")

def test_extract_text_by_page_empty_data():
    pages = extract_text_by_page({})
    assert pages == {}

def test_extract_headers_by_page_empty_data():
    headers = extract_headers_by_page({"pages": {}})
    assert headers == {}
