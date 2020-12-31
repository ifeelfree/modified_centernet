from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from types import MethodType

#from torchviz import make_dot


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)

  # viewer the network architecture
  if False:
    from my_lib.path.path_manager import OutputPathManager
    path_manger = OutputPathManager()
    pdf_file_name = path_manger.create_output_file("model_architecture", "model.pdf")
    from my_lib.visualization.model_vis import draw_nn_model
    draw_nn_model(model, pdf_file_name, (1, 3, 512, 512))

  # check the input/out
  if False:
      from my_lib.path.path_manager import OutputPathManager
      path_manger = OutputPathManager()
      txt_file_name = path_manger.create_output_file("model_architecture", "model.txt")
      from my_lib.visualization.model_vis import describe_nn_model
      describe_nn_model(model, txt_file_name, (3, 512, 512))

  if False:
    tmp_image = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, tmp_image, "/home/jianglin/workspace/sk/education/modified_centernet/CenterNet/my_output/model_architecture/model.onnx")


  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  # ctdet --exp_id coco_dla_2x --batch_size 2  --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 230
  # ctdet --exp_id coco_resdcn18 --arch resdcn_18 --batch_size 114 --master_batch 18 --lr 5e-4 --gpus 0 --num_workers 16
  # ctdet --exp_id coco_dla_2x --batch_size 6  --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 20
  opt = opts().parse()
  main(opt)