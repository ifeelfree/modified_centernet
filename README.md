# Modified CenterNet
The purpose of this project is to improve CenterNet so that it can
- train user-defined data much easier
- improve the original codebase 

# Package Installation
## Repository 
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [Deformable Convolutional Networks V2 with Pytorch 1.X](https://github.com/CharlesShang/DCNv2)
- [Deformable Convolutional Networks V2 with Pytorch 1.0](https://github.com/lbin/DCNv2/tree/pytorch_1.6)
- [COCO API](https://github.com/cocodataset/cocoapi)
- [Local COCO Viewer](https://github.com/trsvchn/coco-viewer)
- [COCO_Image_Viewer]()


import torch.onnx
  tmp_image = torch.randn(1, 3, 512, 512)
  torch.onnx.export(model, tmp_image, "majianglin.onnx")