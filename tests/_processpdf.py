# test_processpdf.py
import pytest
import os
import sys

# Get the correct path to docling-core
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# docling-core is in the same directory as the script
docling_core_path = os.path.join(current_dir, 'docling-core')


# Add to Python path
sys.path.insert(0, docling_core_path)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from document_processing.utilities import get_file_name_without_ext
from pathlib import Path

@pytest.skip(reason="Skipping test_processpdf.py")

# Additional fixtures
@pytest.fixture
def document_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

# Test document converter initialization

def test_document_converter_initialization(document_converter):
    assert isinstance(document_converter, DocumentConverter)
    assert document_converter.format_to_options is not None

# Test document conversion
def test_document_conversion(document_converter, valid_pdf_path):
    result = document_converter.convert(str(valid_pdf_path))
    assert result is not None
    assert hasattr(result, 'document')
    assert result.document is not None

# Test conversion with invalid file
def test_document_conversion_invalid_file(document_converter, invalid_pdf_path):
    with pytest.raises(Exception):
        document_converter.convert(str(invalid_pdf_path))

# Test file name extraction
def test_get_file_name_without_ext_valid_path():
    test_path = "../pdfs/test_document.pdf"
    result = get_file_name_without_ext(test_path)
    assert result == "test_document"

def test_get_file_name_without_ext_no_extension():
    test_path = "../pdfs/test_document"
    result = get_file_name_without_ext(test_path)
    assert result == "test_document"

def test_get_file_name_without_ext_invalid_path():
    test_path = None
    result = get_file_name_without_ext(test_path)
    assert result is None

def test_get_file_name_without_ext_empty_string():
    test_path = ""
    result = get_file_name_without_ext(test_path)
    assert result is None


