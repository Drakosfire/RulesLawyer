from PyPDF2 import PdfReader
import os
import math

def estimate_conversion_time(num_pages):
    """Estimate the total conversion time based on ~5 seconds per page"""
    return num_pages * 5

def format_time(seconds):
    """Convert seconds to a human-readable format"""
    minutes = math.floor(seconds / 60)
    remaining_seconds = seconds % 60
    return f"{minutes} minutes and {remaining_seconds:.0f} seconds"

def check_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False
    
    try:
        with open(file_path, 'rb') as file:
            PdfReader(file)
        print(f"PDF '{file_path}' can be opened successfully.")
        return True
    except Exception as e:
        print(f"Error opening PDF '{file_path}': {str(e)}")
        return False

def check_pdf_details(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            num_pages = len(pdf.pages)
            print(f"Number of pages: {num_pages}")
            print(f"PDF Version: {pdf.pdf_header}")
            print(f"File size: {os.path.getsize(file_path)} bytes")
            if pdf.metadata:
                print("Metadata:")
                for key, value in pdf.metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("No metadata available")
            return num_pages
    except Exception as e:
        print(f"Error checking PDF details: {str(e)}")
        return None