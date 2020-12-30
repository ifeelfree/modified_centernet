import cv2

class GrayToColor(object):
    """
    This class is used to transform gray-scale image to color image

    The following references related to this classes are:
    - https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    - https://stackoverflow.com/questions/3016283/create-a-color-generator-from-given-colormap-in-matplotlib
    - https://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/

    We finally use the last reference

    """
    COLOR_MAP={
        'rainbow':cv2.COLORMAP_RAINBOW,
        'hot':cv2.COLORMAP_HOT,
    } # https://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
    # for color range
    def __init__(self, color_map_name=None):
        """
        intilization

        :param color_map color mapping function
        """
        import matplotlib.pyplot as plt
        if color_map_name is None:
            color_map_name = 'hot'

        self.color_map_name_ = color_map_name


    def __call__(self, *args, **kwargs):
        """
        apply psuducoloring on the given gray image(s)
        """
        color_img_list = []
        if self.color_map_name_ == 'truecolor':
            pass
        else:
            color_map_mode = self.COLOR_MAP[self.color_map_name_]
        for arg in args:
            from my_lib.visualization.image_vis import normalized_255
            arg = normalized_255(arg)
            if self.color_map_name_ == 'truecolor':
                import numpy as np
                color_img = np.dstack((arg, arg, arg))
            else:
                color_img = cv2.applyColorMap(arg, color_map_mode)
                color_img = color_img[:, :, [2, 1, 0]]
            color_img_list.append(color_img)



        if len(args) == 1:
            return color_img_list[0]

        return color_img_list




if __name__ == "__main__":
    color_transformer = GrayToColor()
    from my_lib.path.demo_data_manager import DemoDataManager
    data_manger = DemoDataManager()
    gray = data_manger.gray_image()
    color = GrayToColor()(gray)
    from my_lib.visualization.image_vis import show_color_image, \
        show_grey_image, \
        show_multiple_images
  #  show_color_image(color)
  #  show_grey_image(gray)
    show_multiple_images(color, gray)


