import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def preprocessing(img):
    """
    This function is to make sure that after preporcessing,
    the image can be used for illustration by plt.imshow. It will
    go through the following operations:
       1) normalized_255: change the image data format to uint8
       2) revert_torch_color: change the color format to [row, col, bands]
    """
    opertion_list = (normalized_255, revert_torch_color)
    for opt in opertion_list:
        img = opt(img)
    return img

def revert_torch_color(img):
    """
    This function is used to make sure the color image format is
    [row, col, bands]
    """
    ndim = img.ndim
    if ndim == 3:
        ar = img.shape
        band_index = ar.index(min(ar))
        if band_index == 0:
            img = np.transpose(img, [1, 2, 0])

    return img


def normalized_255(img):
    """
    this function is used to normlized the image to [0, 255], and the image type
    is np.uint8

    :params img input image
    :return an normalized image whose data type is uint8, min value is 0,
            and max value is 255
    """

    # if it is tensor
    if torch.is_tensor(img):
        img = img.numpy()

    # if it is uint8
    if img.dtype == np.uint8:
        return img

    max_value = np.max(img)
    min_value = np.min(img)

    if max_value-min_value==0:
        pass
    else:
        img = (img-min_value)/(max_value-min_value)*255.8
    img = np.uint8(img)
    return img



def show_grey_image(img):
    """
    thif function is used to show the gray-scale image

    """
    img = preprocessing(img)
    fig, ax = plt.subplots(1, 1)
    ax_img = ax.imshow(img)
    if np.max(img)<=1: # psuducoloring is valid only when input is lower than 1
        fig.colorbar(ax_img)
    ax.set_axis_off()
    plt.show()



def show_color_image(img, **kwargs):
    """
    this function is used to show the image

    :params img input image
    :params kwargs
             kwargs['box']: [(left, bottom, width, height),
                             (left, bottom, width, height)]
             #
             #       ----------------------->>>
             #       |
             #       |  (left, bottom)
             #       |
             #       V
             #       V
    """

    img = preprocessing(img)
    # import matplotlib
    # matplotlib.use("Qt4Agg")




    fig, ax = plt.subplots(1,1)
    ax.imshow(img)

    if kwargs:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon
        polygons = []
        color = []
        if 'box' in kwargs.keys():
            window_list = kwargs['box']
            for (bbox_x, bbox_y, bbox_w, bbox_h) in window_list:
                # generate box
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4, 2))
                polygons.append(Polygon(np_poly))
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                color.append(c)
        if polygons:
            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)


    ax.set_axis_off()
    plt.show()

def show_multiple_images(*args, **kwargs):
    """
    The purpose of this function is to show multiple images (color or gray)
    """
    # check image number
    img_num = len(args)
    plot_shape=(1, img_num)

    fig = plt.figure()

    for index, img in enumerate(args):
        ax = fig.add_subplot(plot_shape[0], plot_shape[1], index+1)
        ax.axis('off')
        img = preprocessing(img)
        ax.imshow(img)

    plt.show()







