#docling_pdf.py
#This script uses Docling to convert a PDF to a JSON file.
#It then uses OpenAI to summarize each page and the entire document.
#It then saves the summarized JSON to a file.
# Instructions: 
# 1. Update the source variable to the path of the PDF file you want to convert.
# 2. Run the script.
# 3. Check the output in the output folder.
# 4. Then run jsontomd.py to create the markdown file.
# 5. Then run jsonToEmbeddings.py to create the embeddings.  
# 6. Point the app.py to the enhanced JSON file.
# 7. Run app.py to start the gradio web app.

import time
import os
import json
import sys
from PyPDF2 import PdfReader
import logging
from openai import OpenAI
from tqdm import tqdm
import math
from document_processing.pdf_utils import check_pdf, check_pdf_details, estimate_conversion_time, format_time
from document_processing.json_utils import save_to_file, load_json, extract_text_by_page, save_enhanced_json, page_enhanced_markdown_export
from document_processing.summarizer import summarize_page, summarize_document
from document_processing.pdf_processor import process_pdf_file
from document_processing.utilities import get_file_name_without_ext
from document_processing.embedding_generator import process_document

# Get the correct path to docling-core
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# docling-core is in the same directory as the script
docling_core_path = os.path.join(current_dir, 'docling-core')

# Add to Python path
sys.path.insert(0, docling_core_path)

# Now try the import
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import DocumentLimits
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import logging


class PDFToEmbeddings:
    def __init__(self, source):
        self.source = source
        self.client = OpenAI()
        self.file_name_without_ext = get_file_name_without_ext(source)
        self.json_file = f"./output/{self.file_name_without_ext}_output.json"
        self.enhanced_json_file = f"./output/{self.file_name_without_ext}_enhanced_output.json"
        self.markdown_file = f"./output/{self.file_name_without_ext}_enhanced_output.md"
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.pages = check_pdf_details(self.source)

    def initialize_converter(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        step1_time = time.time()
        print(f"Step 1 (Initialize Converter): {step1_time - self.last_step_time:.2f} seconds")
        print(f"Cumulative time: {step1_time - self.start_time:.2f} seconds")
        self.last_step_time = step1_time

    def convert_pdf(self):
        print("Document Limits:")
        print(DocumentLimits())

        if check_pdf(self.source):
            num_pages = check_pdf_details(self.source)
            if num_pages:
                estimated_time = estimate_conversion_time(num_pages)
                print(f"\nEstimated conversion time: {format_time(estimated_time)}")
                print("Starting conversion...\n")
                
                with tqdm(total=100, desc="Converting PDF", unit="%") as pbar:
                    try:
                        self.result = process_pdf_file(self.source)
                        pbar.update(100)
                        step2_time = time.time()
                        print(f"\nStep 2 (Convert): {step2_time - self.last_step_time:.2f} seconds")
                        print(f"Cumulative time: {step2_time - self.start_time:.2f} seconds")
                        self.last_step_time = step2_time
                    except Exception as e:
                        print(f"Conversion failed with error: {str(e)}")
                        print("Traceback:")
                        import traceback
                        traceback.print_exc()
        else:
            print("PDF check failed. Conversion aborted.")

    def process_json(self):
        json_output = self.result.document.export_to_dict()
        formatted_json = json.dumps(json_output, indent=2, ensure_ascii=False)
        save_to_file(formatted_json, self.json_file)
        data = load_json(self.json_file)

        pages = extract_text_by_page(data)

        for page in pages:
            page_text = "\n".join([
                pages[page]['text_entries'][entry]['text']
                for entry in pages[page]['text_entries']
            ])
            pages[page]['summary'] = summarize_page(page_text)

        all_summaries = "\n".join([pages[page]['summary'] for page in pages])
        output = summarize_document(all_summaries)
        data['document_summary'] = output

        save_enhanced_json(data, pages, self.enhanced_json_file)
        print(f"JSON enhanced with pages object. Saved to {self.enhanced_json_file}")
        step5_time = time.time()
        print(f"Step 5 (JSON): {step5_time - self.last_step_time:.2f} seconds")
        print(f"Cumulative time: {step5_time - self.start_time:.2f} seconds")
    
    def export_to_markdown(self):
        page_enhanced_markdown_export(self.enhanced_json_file)

    def create_embeddings(self):
        embeddings_df = process_document(self.markdown_file, self.enhanced_json_file)
        embeddings_df.to_csv(f"./output/{self.file_name_without_ext}_embeddings.csv", index=False)

    def run(self):
        self.initialize_converter()
        print("Converting PDF...")
        self.convert_pdf()
        print("Processing JSON...")
        self.process_json()
        print("Exporting to markdown...")
        self.export_to_markdown()
        print("Creating embeddings...")
        self.create_embeddings()

        total_time = time.time() - self.start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")

        try:
            if self.pages :
                print(f"Total time per page: {total_time / self.pages:.2f} seconds")
            else:
                print("Could not calculate time per page: no pages were processed")
        except NameError:
            print("Could not calculate time per page: conversion process did not complete")

if __name__ == "__main__":
    source = "./pdfs/test_document.pdf"
    pdf_to_embeddings = PDFToEmbeddings(source)
    pdf_to_embeddings.run()

