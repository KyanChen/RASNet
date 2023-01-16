import os
import time

import torch
if os.name == 'nt':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from training_hyper import Trainer as Trainer_Hyper
from training_siren import Trainer as Trainer_Siren  # siren, siren_residual
from training_mlp_pe import Trainer as Trainer_mlp
from training_siren_multi_loss import Trainer as Trainer_Multi_Loss
from datasets import INRDataset, get_dataloader


trainer = 'Trainer_Siren'
pattern = 'train'
batch_size = 256
img_size = (28, 28)
# 0：不多尺度训练
# 1: 随机多尺度训练
# 2: 分别按size分尺度
is_train_all_size = 2
# path_prefix = r'D:\GID\RGB_15_train'
# path_prefix = r'H:\DataSet\SceneCls\UCMerced_LandUse\UCMerced_LandUse\Images'
path_prefix = 'RGB_15_train'
# path_prefix = 'UCMerced_LandUse/Images'
max_epoch = int(5e4)
exp = 'EXPWo'
model_dir = f'models/{exp}'
result_dir = f'results/{exp}'
load_checkpoint = f'models/{exp}/300.pth'
# img_file_list = 'data_list/GID/N600/all_list_56_1.txt'
#img_file_list = 'data_list/GID/N600/val_list_112.txt'
img_file_list = 'data_list/GID/N600/fit_list_56_112_224_split.txt'

# img_file_list = 'data_list/GID/all_img_list_112_3class.txt'
# img_file_list = 'data_list/UC/all_img_list.txt'
# img_file_list = 'data_list/UC/val_list.txt'
# pre_train_resnet = 'models/EXP20220501_5.pth'
pre_train_resnet = None
is_BN = False

time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
result_py_dir = result_dir+'/'+ f'{pattern}_{is_train_all_size}_{img_size[0]}_{time_str}'
os.makedirs(result_py_dir, exist_ok=True)
os.system(f'cp *.py {result_py_dir}')
if __name__ == '__main__':
    shuffle = True
    is_test = False
    if pattern == 'val':
        shuffle = False
        is_test = True

    data_set = INRDataset(
        path_prefix, img_file_list, is_load_all=True, 
        img_size=img_size, is_test=is_test,
        is_train_all_size=is_train_all_size)
    data_loader = get_dataloader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    if trainer == 'Trainer_Hyper':
        trainer = Trainer_Hyper(
            data_loader=data_loader, img_size=img_size,
            max_epoch=max_epoch, model_dir=model_dir, result_dir=result_dir,
            load_checkpoint=load_checkpoint
        )
    elif trainer == 'Trainer_Siren':
        trainer = Trainer_Siren(
            data_loader=data_loader, img_size=img_size,
            max_epoch=max_epoch, model_dir=model_dir, result_dir=result_dir,
            load_checkpoint=load_checkpoint, is_train_all_size=is_train_all_size,
            pattern=pattern, is_BN=is_BN
        )
    elif trainer == 'Trainer_mlp':
        trainer = Trainer_mlp(
            data_loader=data_loader, img_size=img_size,
            max_epoch=max_epoch, model_dir=model_dir, result_dir=result_dir,
            load_checkpoint=load_checkpoint
        )
    elif trainer == 'Trainer_Multi_Loss':
        trainer = Trainer_Multi_Loss(
            data_loader=data_loader, img_size=img_size,
            max_epoch=max_epoch, model_dir=model_dir, result_dir=result_dir,
            load_checkpoint=load_checkpoint, is_train_all_size=is_train_all_size,
            pattern=pattern, pre_train_resnet=pre_train_resnet
        )

    if pattern == 'train':
        trainer.train()
    elif pattern == 'val':
        trainer.val()
    elif pattern == 'val_batch':
        trainer.val_batch()


# def collate_fn(batch):
#     inputs = []
#     targets = []
#     for data in batch:
#         input, target = data
#         if input is not None and target is not None:
#             inputs.append(input)
#             targets.append(target)
#     inputs = torch.stack(inputs)
#     targets = torch.stack(targets)
#     return inputs, targets

