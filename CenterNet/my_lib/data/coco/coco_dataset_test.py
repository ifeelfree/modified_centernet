
if __name__ == "__main__":
    from my_lib.data.coco.coco_dataset import CocoDataset, CocoPathManager
    coco_path_manager = CocoPathManager()
    data_set_obj = CocoDataset(coco_path_manager, "train")
    from my_lib.visualization.image_vis import show_color_image

    for index in [400]:  # 100, 400
        img, annot = data_set_obj[index]
        cat_list = annot['category_id_list'].tolist()
        print(data_set_obj.obtain_categories(cat_list))
        box_dict = {}
        box_dict['box'] = annot['box_list']
        show_color_image(img, **box_dict)
