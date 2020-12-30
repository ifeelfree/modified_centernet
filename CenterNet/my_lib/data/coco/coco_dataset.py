import os
import torch
import torch.utils.data
import torchvision
import cv2
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

class CocoPathManager(object):
    """
    This class is used to manage path to CoCo dataset, and
    Coco dataset follows the following path protocal:

    ├── annotations
        ├── captions_train2017.json
        ├── captions_val2017.json
        ├── image_info_test-dev2017.json
        ├── instances_train2017.json
        ├── instances_val2017.json
        ├── person_keypoints_train2017.json
        └── person_keypoints_val2017.json
    ├── test2017
    ├── train2017
    └── val2017

    """
    def __init__(self, given_path=None):
        """
        initialize the coco data path manager

        :params given_path the path to the coco dataset

        """
        if given_path is None:
            import os
            import pathlib
            given_path = pathlib.Path(os.environ['CENTERNET_ROOT'])/'CenterNet'/"data"/"coco"
        assert given_path.exists()
        self.coco_path_ = given_path

    def get_annotation_json(self, sep_type, task_type):
        """
        obtain the annotation json file depending on the given
        task type and data seperation type

        :params sep_type 'train', 'val'
        :params task_type 'detection'
        :return the json file
        """
        assert task_type == "detection"
        assert sep_type in ['train', 'val']

        file_name = f"instances_{sep_type}2017.json"
        file_name = self.coco_path_/"annotations"/file_name
        assert file_name.exists()
        return file_name

    def get_image_directory(self, sep_type):
        """
        obtain the image directory

        :param sep_type data seperation method
        :return the image directory 
        """
        assert sep_type in ['train', 'val']
        dir_name = sep_type+'2017'
        dir_name = self.coco_path_/ dir_name
        assert dir_name.exists()
        return dir_name





class CocoDataset(torch.utils.data.Dataset):
    """
    The purpose of this class is to have a wrapper for Coco dataset
    This class is inspired by the article (https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5)

    """
    def __init__(self, path_manager: CocoPathManager,  sep_type: str,
                 task_type='detection',
                 transform=None):
        """
        initialize the coco data set with its path manager

        :param
        """
        annotation = path_manager.get_annotation_json(sep_type, task_type)
        self.coco_ = COCO(annotation)
        self.ids_ = list(sorted(self.coco_.imgs.keys()))
        self.transform_ = transform
        self.image_dir_ = path_manager.get_image_directory(sep_type)

    def obtain_categories(self, cat_index_list):
        """
        This function is used to obtain the categories that the given category list contains

        :param cat_index_list category index list
        :return category string list
        """
        cats_dict_arr = self.coco_.loadCats(cat_index_list)
        cat_list = [ t['name'] for t in cats_dict_arr]
        return cat_list

    def dataset_summary(self):
        """
        This function is used to print all the information of the dataset
        """
        cat_info = self._category_info()
        img_info = self._image_info()
        return {**cat_info, **img_info}

    def _category_info(self):
        """
        return the dataset's category information
        """
        cat_name_list = set()
        super_cat_name_list = set()
        for k, v in self.coco_.cats.items():
            cat_name_list.add(v['name'])
            super_cat_name_list.add(v['supercategory'])

        return {
            'class':list(cat_name_list),
            'class_number':len(cat_name_list),
            'superclass':list(super_cat_name_list),
            'superclass_number': len(super_cat_name_list),
        }

    def _image_info(self):
        """
        return the image's information
        """
        return {'image_number': len(self.coco_.imgs)}


    def get_segmentation_mask(self, index, b_instance=False):
        """
        this function is used to get the segmentation mask
        """
        img_id = self.ids_[index]
        ann_ids =self.coco_.getAnnIds(imgIds=img_id)
        img_dict = self.coco_.imgs[img_id]
        img_height, img_width = img_dict['height'], img_dict['width']
        coco_annotation = self.coco_.loadAnns(ann_ids)
        mask = np.zeros((img_height, img_width), np.uint8)
        for index, ann in enumerate(coco_annotation):
            cat_id = int(ann['category_id'])
            if b_instance is False:
                mask = np.maximum(self.coco_.annToMask(ann)*cat_id, mask)
            else:
                mask = np.maximum(self.coco_.annToMask(ann)*(index+1), mask)
        return mask


    def __getitem__(self, index):
        # Own coco file
        coco = self.coco_
        # Image ID
        img_id = self.ids_[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)

        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        #img = Image.open(str(self.image_dir_/path))
        # cv2 read image as bgr
        img = cv2.imread(str(self.image_dir_/path))
        # transform bgr to rgb https://note.nkmk.me/en/python-opencv-bgr-rgb-cvtcolor/
        img = img[:,:,[2,1,0]]

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]

        boxes = []
        cats = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            width = coco_annotation[i]['bbox'][2]
            height = coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, width, height])
            cats.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        #  crowd list
        crowd_list = []
        for i in range(num_objs):
            ann = coco_annotation[i]
            crowd_list.append(ann['iscrowd'])




        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # box categories
        cats = torch.as_tensor(cats, dtype=torch.int)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["box_list"] = boxes
        my_annotation["image_id"] = img_id
        my_annotation["area_list"] = areas
        my_annotation["category_id_list"] = cats
        my_annotation["crowd_list"] = crowd_list


        if self.transform_ is not None:
            img = self.transform_(img)
        else:
            img = torch.from_numpy(img)


        return img, my_annotation

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    from my_lib.data.coco.coco_dataset import CocoDataset, CocoPathManager
    from my_lib.visualization.image_vis import show_single_image

    coco_path_manager = CocoPathManager()
    data_set_obj = CocoDataset(coco_path_manager, "train")









