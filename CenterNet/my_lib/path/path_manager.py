import pathlib
import os


def retrieve_project_path():
    """
    get the directory of the project CenterNet
    """
    path_dir = pathlib.Path(__file__).resolve().parent
    lib_dir = path_dir.parent
    proj_dir = lib_dir.parent
    return proj_dir

class OutputPathManager(object):
    """
    This class is used to manage path related to our experiments
    """
    def __init__(self):
        """
        initialize project path, output path
        """

        proj_dir = retrieve_project_path()
        output_dir = proj_dir / 'my_output'
        assert output_dir.exists()
        self.project_dir_ = proj_dir
        self.output_dir_ = output_dir

    @property
    def get_output_directory(self):
        """
        get output directory
        """
        return self.output_dir_

    def create_output_file(self, *short_file_path):
        """
        create file in the output directory
        """
        tmp_path = self.output_dir_.joinpath(*short_file_path)
        if tmp_path.exists():
            pass
        else:
            my_dir = tmp_path
            tmp_suffix = my_dir.suffix
            if tmp_suffix:
                my_dir = tmp_path.parent
            if my_dir.exists():
                pass
            else:
                os.makedirs(my_dir)

        return tmp_path


