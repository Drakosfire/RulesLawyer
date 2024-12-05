import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "test_data"

@pytest.fixture
def valid_pdf_path(test_data_dir):
    path = test_data_dir / "valid_test.pdf"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return path

@pytest.fixture
def invalid_pdf_path(test_data_dir):
    return test_data_dir / "nonexistent.pdf" 