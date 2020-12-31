"""
This file contains functions
"""

def prepare_filepath_for_writing(file_name):
    """
    before writing to the file, check whehter the path exists, and if
    it does not exist, create the directory that contains the file.

    :params file_name file name (for writing)

    """
    import pathlib
    parent_path = pathlib.Path(file_name).absolute().parent
    if parent_path.exists():
        pass
    else:
        import os
        os.makedirs(str(parent_path))

