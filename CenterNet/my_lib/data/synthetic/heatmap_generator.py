import numbers
import numpy as np

class HeatmapGenerator(object):
    """
    This class is used to generate object heatmaps
    """
    def __init__(self, method='msra'):
        """
        initilisize the heat map generator

        :params method
        """
        self.method_ = method


    def msra_gaussian(self, heatmap: np.array, center, sigma):
        """
        generate MSRA gaussian heatmap

        :param heatmap in and out
        :param center (x,y)
        :param sigma (x,y) or a number

        """

        if sigma[0] == sigma[1]:
            from my_lib.data.synthetic.heatmap_generator_fun import draw_msra_gaussian
            return draw_msra_gaussian(heatmap, center, sigma[0])
        else:
            assert 0

    def visualize_heatmap(self, input_image:np.array, heatmap:np.array):
        """
        This function is to enhance visualization,

        :param input_image input image /gray/color
        :param hetamp the heatmap
        """
        # step 0: resize image if it is necessary
        import cv2
        from my_lib.visualization.image_vis import preprocessing
        input_image  = preprocessing(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)


        input_image = cv2.resize(input_image, (heatmap.shape[1], heatmap.shape[0]))



        # step 1: collect image as background image
        from my_lib.data.image.gray2color import GrayToColor
        if input_image.ndim == 3:
            pass
        else:
            input_image = GrayToColor(color_map_name='truecolor')(input_image)

        # step 2: hetamp color
        from my_lib.data.image.gray2color import GrayToColor
        color_heatmap = GrayToColor(color_map_name='hot')(heatmap)

        # step 3: mask the color heatmap
        mask = np.ones_like(heatmap, np.uint8)
        mask[heatmap > 0] = 0
        color_heatmap[mask == 1, :] = 0 # heatmap background will be empty

        # step 4: add color heatmap to the input image
        input_image = input_image.astype(np.float) + color_heatmap.astype(np.float)

        # step 5: visualize it
        from my_lib.visualization.image_vis import show_single_image
        show_single_image(input_image)








    def generate_heatmap(self, heatmap: np.array, center, sigma, **kwargs):
        """
        This function is used to generate heatmap

        :params heatmap heatmap, and its range is between 0 and 1 (float)
        :params center (x, y)
        :sigma  sigma  (x, y) or a number
        :param kwargs
                return matrix
                kwargs['trans']: None [0, 1] float
                                 'color' color image
                                 'gray' gray image

        """
        output_heatmap = None
        if isinstance(sigma, numbers.Number):
            sigma = [sigma, sigma]

        if self.method_ == "msra":
            output_heatmap =  self.msra_gaussian(heatmap, center, sigma)
        else:
            assert 0


        if kwargs.get('trans') is None:
            pass
        elif kwargs.get('trans') =='gray':
            from my_lib.visualization.image_vis import normalized_255
            output_heatmap = normalized_255(output_heatmap)
        elif kwargs.get('trans') =='color':
            from my_lib.data.image.gray2color import GrayToColor
            output_heatmap = GrayToColor()(output_heatmap)
        else:
            assert 0

        return output_heatmap




if __name__ == "__main__":
    from my_lib.path.demo_data_manager import DemoDataManager
    demo_path_manger = DemoDataManager()
    img_gray = demo_path_manger.gray_image()
    row, col = img_gray.shape[0], img_gray.shape[1]

    from my_lib.data.synthetic.heatmap_generator import HeatmapGenerator
    generator = HeatmapGenerator()
    heatmap = np.zeros((row,col), np.float)
    generator.generate_heatmap(heatmap,(int(col*1.0/4),int(row*1.0/2)), 13)

    generator.visualize_heatmap(img_gray, heatmap)








