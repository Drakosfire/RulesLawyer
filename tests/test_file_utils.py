import pytest
from pathlib import Path
import json
import logging
from document_processing.file_utils import generate_file_names, save_json, load_json

# skip tests for now
pytestmark = pytest.mark.skip(reason="Skipping file_utils tests")

# Fixtures
@pytest.fixtur
def temp_output_dir(tmp_path):
    """Creates a temporary output directory for testing"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing save/load operations"""
    return {
        "name": "Test Document",
        "pages": 5,
        "metadata": {
            "author": "Test Author",
            "date": "2024-03-19"
        }
    }

# Test generate_file_names
@pytest.mark.dev
def test_generate_file_names_with_valid_source():
    source = "path/to/test.pdf"
    json_file, enhanced_file = generate_file_names(source)
    assert json_file == "../output/test_output.json"
    assert enhanced_file == "../output/test_enhanced_output.json"

@pytest.mark.dev
def test_generate_file_names_with_custom_output_dir(temp_output_dir):
    source = "path/to/test.pdf"
    json_file, enhanced_file = generate_file_names(source, str(temp_output_dir))
    assert json_file == f"{temp_output_dir}/test_output.json"
    assert enhanced_file == f"{temp_output_dir}/test_enhanced_output.json"

@pytest.mark.dev
def test_generate_file_names_with_empty_source():
    json_file, enhanced_file = generate_file_names("")
    assert json_file == "../output/default_output.json"
    assert enhanced_file == "../output/default_enhanced_output.json"

@pytest.mark.dev
def test_generate_file_names_with_none_source():
    json_file, enhanced_file = generate_file_names(None)
    assert json_file == "../output/default_output.json"
    assert enhanced_file == "../output/default_enhanced_output.json"

# Test save_json and load_json
@pytest.mark.dev
def test_save_and_load_json(temp_output_dir, sample_json_data):
    test_file = temp_output_dir / "test.json"
    
    # Test saving
    save_json(sample_json_data, str(test_file))
    assert test_file.exists()
    
    # Test loading
    loaded_data = load_json(str(test_file))
    assert loaded_data == sample_json_data

@pytest.mark.dev
def test_save_json_with_unicode(temp_output_dir):
    unicode_data = {"text": "Hello 世界"}
    test_file = temp_output_dir / "unicode.json"
    
    save_json(unicode_data, str(test_file))
    loaded_data = load_json(str(test_file))
    assert loaded_data["text"] == "Hello 世界"

@pytest.mark.dev
def test_load_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_json("nonexistent.json")

@pytest.mark.dev
def test_save_json_invalid_path(temp_output_dir):
    invalid_path = temp_output_dir / "invalid" / "test.json"
    with pytest.raises(FileNotFoundError):
        save_json({"test": "data"}, str(invalid_path))

# Test error logging
@pytest.mark.dev
def test_generate_file_names_logs_warning_for_invalid_source(caplog):
    with caplog.at_level(logging.WARNING):
        generate_file_names(None)
    assert "Using default file names" in caplog.text

@pytest.mark.dev
def test_generate_file_names_logs_error_for_exception(caplog):
    with caplog.at_level(logging.ERROR):
        # Pass an object that will cause an exception when processed
        generate_file_names(object())
    assert "Unexpected error occurred" in caplog.text
