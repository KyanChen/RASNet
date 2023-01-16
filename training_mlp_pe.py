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
from siren_modulation import Siren_Modulation
from mlp_pe import MLP_PE


class Trainer:
    def __init__(self,
                 data_loader, img_size=(32, 32),
                 num_modulation=512, max_epoch=50000, print_freq=10, device='cuda',
                 model_dir='tmp', load_checkpoint=''
                 ):
        self.data_loader = data_loader
        self.device = device
        self.print_interval = print_freq
        # 需要修改
        self.mlp_pe = MLP_PE(
            inner_layers=6, in_channels=2, out_channels=3, base_channels=512,
            num_modulation=num_modulation, bias=True, expansions=[1]
            )
        self.para = torch.nn.Parameter(torch.zeros(1, num_modulation))

        self.img_size = img_size
        self.coordinates = to_coordinates(self.img_size)
        self.coordinates = self.coordinates.to(device)

        self.num_modulation = num_modulation
        
        self.optimizer_w = torch.optim.AdamW(self.mlp_pe.parameters(), lr=1e-5)
        self.optimizer_b = torch.optim.SGD([self.para], lr=1e-2)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, T_max=200, eta_min=1e-7)
        self.loss_func = torch.nn.MSELoss()

        self.max_epoch = max_epoch
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists(load_checkpoint):
            print(f'load checkpoint from {load_checkpoint}')
            states = torch.load(load_checkpoint)
            self.mlp_pe.load_state_dict(states['mlp_pe'])
        else:
            print('train from scratch ......')
        self.mlp_pe = self.mlp_pe.to(device)

    def train(self):
        for i_epoch in range(self.max_epoch):
            if i_epoch % 50 == 0:
                state_dict = {
                    'mlp_pe': self.mlp_pe.state_dict()
                }
                torch.save(state_dict, f'{self.model_dir}/{i_epoch}.pth')
                torch.save(state_dict, f'{self.model_dir}/latest.pth')

            for batch_id, data in enumerate(self.data_loader):
                img = data.to(self.device)  # B C H W
                img = rearrange(img, 'B C H W -> B (H W) C')

                modulations_tmp = []
                self.mlp_pe.freeze_model_w_b()
                
                for i_batch in range(img.size(0)):
                    self.para.data = torch.zeros_like(self.para).to(self.device)
                    self.para.requires_grad = True
                    for i_inner_iter in range(6):
                        predicted = self.mlp_pe(self.coordinates, self.para)
                        loss = self.loss_func(predicted, img[i_batch])
                        self.optimizer_b.zero_grad()
                        loss.backward()
                        self.optimizer_b.step()
                    modulations_tmp.append(self.para.data)

                self.para.requires_grad = False
                self.mlp_pe.train_model_w_b()
                losses = []
                psnres = []
                for i_batch in range(img.size(0)):
                    modulation = modulations_tmp[i_batch]
                    predicted = self.mlp_pe(self.coordinates, modulation)
                    loss = self.loss_func(predicted, img[i_batch])
                    psnres.append(get_clamped_psnr(predicted, img[i_batch]))
                    losses.append(loss)
                losses = sum(losses)
                self.optimizer_w.zero_grad()
                losses.backward()
                self.optimizer_w.step()
                if batch_id % self.print_interval == 0:
                    print(f'{i_epoch}: {batch_id}/{len(self.data_loader)}:  {np.mean(psnres)}')
            
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


