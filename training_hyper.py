import os

import numpy as np
import torch
import tqdm
from collections import OrderedDict
from util import get_clamped_psnr, to_coordinates
from skimage import io
from torchvision import transforms
from einops import repeat, rearrange
import json
from torchvision.utils import save_image
from torchvision.models.resnet import resnet18 as _resnet18
from siren_res import Siren_Res


class Trainer:
    def __init__(self,
                 data_loader, img_size=(32, 32),
                 num_modulation=512, max_epoch=50000, print_freq=10, device='cuda',
                 model_dir='tmp', load_checkpoint=''
                 ):
        self.data_loader = data_loader
        self.device = device
        self.print_interval = print_freq
        self.siren = Siren_Res(inner_layers=6, in_channels=2, out_channels=3, base_channels=512,
                               num_modulation=num_modulation, bias=True, expansions=[1])
        self.img_size = img_size
        self.coordinates = to_coordinates(self.img_size)
        self.coordinates = self.coordinates.to(device)

        self.num_modulation = num_modulation
        self.resnet = _resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(512, self.num_modulation, bias=True)

        self.optimizer = torch.optim.AdamW([{'params': self.siren.parameters()},
                                            {'params': self.resnet.parameters(), 'lr': 1e-5}
                                            ],
                                           lr=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-7)
        self.loss_func = torch.nn.MSELoss()

        self.max_epoch = max_epoch
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists(load_checkpoint):
            print(f'load checkpoint from {load_checkpoint}')
            states = torch.load(load_checkpoint)
            self.siren.load_state_dict(states['siren'])
            self.resnet.load_state_dict(states['resnet'])
        else:
            print('train from scratch ......')
        self.resnet = self.resnet.to(device)
        self.siren = self.siren.to(device)

    def train(self):
        for i_epoch in range(self.max_epoch):
            if i_epoch % 50 == 0:
                state_dict = {
                    'siren': self.siren.state_dict(),
                    'resnet': self.resnet.state_dict()
                }
                torch.save(state_dict, f'{self.model_dir}/{i_epoch}.pth')
                torch.save(state_dict, f'{self.model_dir}/latest.pth')

            for batch_id, data in enumerate(self.data_loader):
                img = data.to(self.device)  # B C H W
                latent_codes = self.resnet(img)  # B C
                coordinates = repeat(self.coordinates, 'N C -> (B N) C', B=latent_codes.size(0))
                latent_codes_repeat = repeat(latent_codes, 'B C -> (B N) C', N=self.coordinates.size(0))
                pred_rgb = self.siren(coordinates, latent_codes_repeat)
                target = rearrange(img, 'B C H W -> (B H W) C')
                loss = self.loss_func(pred_rgb, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch_id % self.print_interval == 0:
                    psnr = get_clamped_psnr(pred_rgb, target)
                    print(f'{i_epoch}: {batch_id}/{len(self.data_loader)}:  {psnr}')

            self.lr_scheduler.step()


    def val(self, result_dir='results'):
        os.makedirs(result_dir, exist_ok=True)
        for idx, file_path in enumerate(self.ori_samples):
            feature = self.features[idx]
            feature = feature.to(self.device)
            self.representation.freeze_model_w_b()
            self.para.requires_grad = True

            best_vals = {'loss': 1e8, 'psnr': 0}
            self.para.data = torch.zeros_like(self.para)
            with tqdm.trange(1000, ncols=100) as t:
                for i in t:
                    predicted = self.representation(self.coordinates, self.para)
                    loss = self.loss_func(predicted, feature)
                    self.optimizer_b.zero_grad()
                    loss.backward()
                    self.optimizer_b.step()
                    psnr = get_clamped_psnr(predicted, feature)

                    log_dict = {'loss': loss.item(),
                                'psnr': psnr,
                                'best_psnr': best_vals['psnr']}
                    t.set_postfix(**log_dict)

                    if loss.item() < best_vals['loss']:
                        best_vals['loss'] = loss.item()
                    if psnr > best_vals['psnr']:
                        best_vals['psnr'] = psnr
                        file_name = os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path)
                        os.makedirs(os.path.dirname(result_dir +'/' + file_name), exist_ok=True)
                        json_dict = {}
                        json_dict[file_name] = self.para.data.cpu().numpy().tolist()
                        json.dump(json_dict, open(result_dir +'/'+file_name+'_wo_train.json', 'w'))
                        img_recon = predicted.reshape(*self.img_size, 3).permute(2, 0, 1).float()
                        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), result_dir +'/'+ file_name+'_wo_train.png')


