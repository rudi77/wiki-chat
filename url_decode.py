import os
from urllib.parse import unquote

import os
import urllib.parse

def sanitize_filename(filename):
    # Decode URL-encoded sequences
    decoded_name = urllib.parse.unquote(filename)
    # Define a list of invalid characters
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    # Replace invalid characters with an underscore or any other placeholder
    for char in invalid_chars:
        decoded_name = decoded_name.replace(char, '_')
    return decoded_name

def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            sanitized_name = sanitize_filename(file)
            if file != sanitized_name:
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, sanitized_name)
                try:
                    os.rename(old_path, new_path)
                    print(f'Renamed: {old_path} -> {new_path}')
                except OSError as e:
                    print(f'Error renaming {old_path}: {e}')

# Example usage
directory_path = r'C:\Users\rudi\source\repos\Rechnungserkennung.wiki.dev'
rename_files_in_directory(directory_path)

