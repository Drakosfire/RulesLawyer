from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from document_processing.utilities import get_file_name_without_ext

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

def process_pdf(pdf_path):
    converter = document_converter()
    result = converter.convert(pdf_path)
    return result

def process_pdf_file(pdf_path):
    file_name = get_file_name_without_ext(pdf_path)
    result = process_pdf(pdf_path)
    return result

