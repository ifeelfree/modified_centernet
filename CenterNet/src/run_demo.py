import os


if __name__ == "__main__":
    demo_image_dir = "../data/coco/demo"
    cmd = 'python demo.py ctdet --demo '+demo_image_dir
    model = "ctdet_coco_dla_2x.pth"
    cmd = cmd+' --load_model ../models/'+model
    os.system(cmd)