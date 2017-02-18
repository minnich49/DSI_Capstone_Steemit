import os

# Check for directory and if not there, create one
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Check for each directory in the directory list
def ensure_directories(dir_list):
    for directory in dir_list:
        ensure_directory(directory)


