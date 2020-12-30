

class DemoDataManager(object):
    """
    This class is used to manged the data for demos
    """
    def __init__(self, b_return_as_path=False):
        """
        initializer

        :params b_return_as_path return the demo data as a path (False)
                                 return the demo data as a numpy.array (True)
        """
        from my_lib.path.path_manager import retrieve_project_path
        self.proj_path_ = retrieve_project_path()
        self.b_return_as_path_ = b_return_as_path


    def gray_image(self, data_name='parrot'):
        """
        gray scale image

        :param data_name gray image name
        """
        img_name = self.proj_path_ / "my_data" / "gray" / str(data_name + '.png')
        assert img_name.exists()
        if self.b_return_as_path_:
            return img_name

        import cv2
        img = cv2.imread(str(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img



    def color_image(self, data_name="car"):
        """
        get the color image

        :param data_name the data name
        """
        img_name = self.proj_path_ / "my_data"/"color" / str(data_name+'.jpg')
        assert img_name.exists()
        if self.b_return_as_path_:
            return img_name

        import cv2
        img = cv2.imread(str(img_name)) #BGR
        img = img[:,:,::-1] #RGB
        return img
