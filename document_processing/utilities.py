import os
import logging

def get_file_name_without_ext(file_path):
    try:
        if not file_path:  # Check for empty string
            return None
        # Extract the base name from the file path
        base_name = os.path.basename(file_path)
        # Remove the file extension
        file_name_without_ext = os.path.splitext(base_name)[0]
        return file_name_without_ext
    except (TypeError, AttributeError) as e:
        logging.error(f"Error processing file path: {e}")
        return None