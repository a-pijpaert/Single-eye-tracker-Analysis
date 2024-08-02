import os
import shutil
import re
from datetime import datetime

def copy_analysis_data(src_root, dest_root):
    # Regular expression pattern to match folders starting with 's' followed by 3 digits
    pattern = re.compile(r'^s\d{3}$')

    for subdir in os.listdir(src_root):
        subdir_path = os.path.join(src_root, subdir)
        if os.path.isdir(subdir_path) and pattern.match(subdir):
            analysis_data_src = os.path.join(subdir_path, 'analysis data')
            if os.path.isdir(analysis_data_src):
                # Construct corresponding destination path
                dest_subdir = os.path.join(dest_root, subdir)
                
                # Create destination directory if it doesn't exist
                os.makedirs(dest_subdir, exist_ok=True)
                
                # Copy the entire 'sxxx' folder to the destination
                shutil.copytree(subdir_path, dest_subdir, dirs_exist_ok=True)
                print(f"Copied '{subdir_path}' to '{dest_subdir}'")
            else:
                print(f"No 'analysis data' folder found in '{subdir_path}'")
        else:
            print(f"'{subdir_path}' does not match the pattern 'sXXX'")

# Example usage
source_directory = '/home/arthur/Projects/AMESMC/data'
destination_directory = '/home/arthur/AUMC/Artikelen,studies/Single eye-tracker validation/Analysis/data'
copy_analysis_data(source_directory, destination_directory)
