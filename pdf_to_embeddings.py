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
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import DocumentLimits
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import json
from PyPDF2 import PdfReader
import logging
from openai import OpenAI
from tqdm import tqdm
import math
from document_processing.pdf_utils import check_pdf, check_pdf_details, estimate_conversion_time, format_time
from document_processing.json_utils import save_to_file, load_json, extract_text_by_page, save_enhanced_json
from document_processing.summarizer import summarize_page, summarize_document
from document_processing.pdf_processor import process_pdf_file
from document_processing.utilities import get_file_name_without_ext
from document_processing.embedding_generator import process_document

client = OpenAI()
# Detailed Debugging 
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger('docling')
# logger.setLevel(logging.DEBUG)

source = "./pdfs/test_document.pdf"  # PDF path or URL

start_time = time.time()
last_step_time = start_time

# Step 1: Initialize DocumentConverter with proper options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False  # adjust as needed
pipeline_options.do_table_structure = True  # adjust as needed

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)
step1_time = time.time()
print(f"Step 1 (Initialize Converter): {step1_time - last_step_time:.2f} seconds")
print(f"Cumulative time: {step1_time - start_time:.2f} seconds")
last_step_time = step1_time

print("Document Limits:")
print(DocumentLimits())

# Before converting, check if the PDF can be opened
if check_pdf(source):
    num_pages = check_pdf_details(source)
    if num_pages:
        estimated_time = estimate_conversion_time(num_pages)
        print(f"\nEstimated conversion time: {format_time(estimated_time)}")
        print("Starting conversion...\n")
        
        # Create progress bar
        with tqdm(total=100, desc="Converting PDF", unit="%") as pbar:
            try:
                result = process_pdf_file(source)
                pbar.update(100)
                step2_time = time.time()
                print(f"\nStep 2 (Convert): {step2_time - last_step_time:.2f} seconds")
                print(f"Cumulative time: {step2_time - start_time:.2f} seconds")
                last_step_time = step2_time
            except Exception as e:
                print(f"Conversion failed with error: {str(e)}")
                print("Traceback:")
                import traceback
                traceback.print_exc()
else:
    print("PDF check failed. Conversion aborted.")


try:
    # Replace the existing code for extracting the file name with this function call
    file_name_without_ext = get_file_name_without_ext(source)
    
    if file_name_without_ext is not None:
        # Use the file name for output files
        json_file = f"../output/{file_name_without_ext}_output.json"
        enhanced_json_file = f"../output/{file_name_without_ext}_enhanced_output.json"
    else:
        # Fallback to a default name if there's an error
        logging.warning("Using default file names due to error in file path processing.")
        json_file = "./output/default_output.json"
        enhanced_json_file = "./output/default_enhanced_output.json"
except Exception as e:
    logging.error(f"Unexpected error occurred: {e}")

# Use the file name for output files
json_file = f"./output/{file_name_without_ext}_output.json"
enhanced_json_file = f"./output/{file_name_without_ext}_enhanced_output.json"

# Use the new export methods
json_output = result.document.export_to_dict()  # Changed from result.render_as_dict()
formatted_json = json.dumps(json_output, indent=2, ensure_ascii=False)
save_to_file(formatted_json, json_file)
# Load JSON
data = load_json(json_file)
# print(data)

# Extract text by page
pages = extract_text_by_page(data)
# print(pages)

# Before summarizing pages, extract text from the new structure
for page in pages:
    # Join only the text values from the text_entries dictionaries
    page_text = "\n".join([
        pages[page]['text_entries'][entry]['text']  # Access the 'text' field of each entry
        for entry in pages[page]['text_entries']
    ])
    pages[page]['summary'] = summarize_page(page_text)

# Add all the summaries to a single string
all_summaries = "\n".join([pages[page]['summary'] for page in pages])

output = summarize_document(all_summaries)

# Add the document summary to the JSON
data['document_summary'] = output

save_enhanced_json(data, pages, enhanced_json_file)

print(f"JSON enhanced with pages object. Saved to {enhanced_json_file}")
step5_time = time.time()
print(f"Step 5 (JSON): {step5_time - last_step_time:.2f} seconds")
print(f"Cumulative time: {step5_time - start_time:.2f} seconds")

# Process the document to create embeddings and save to a CSV file
embeddings_df = process_document(enhanced_json_file)

# Save the DataFrame to a CSV file
embeddings_df.to_csv(f"./output/{file_name_without_ext}_embeddings.csv", index=False)

last_step_time = step5_time

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")

# Add error handling for the per-page calculation
try:
    if pages and len(pages) > 0:
        print(f"Total time per page: {total_time / len(pages):.2f} seconds")
    else:
        print("Could not calculate time per page: no pages were processed")
except NameError:
    print("Could not calculate time per page: conversion process did not complete")

