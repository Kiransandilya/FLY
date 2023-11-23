from modules import *

def unzip(path, file_name):
    zip_file_path = f"{path}/{file_name}.zip"
    extract_to_path = f"{path}/{file_name}"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the target directory
        zip_ref.extractall(extract_to_path)

    print(f"Successfully extracted to {extract_to_path}")