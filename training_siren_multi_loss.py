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
from siren_modulation import Siren_Modulation
from logger import Logger
import pickle
import torch.nn.functional as F

class Trainer:
    def __init__(self,
                 data_loader, img_size=(32, 32),
                 num_modulation=1024, max_epoch=50000, print_freq=5, device='cuda',
                 model_dir='tmp', load_checkpoint='', result_dir=None,
                 is_train_all_size=False, 
                 pattern='train', pre_train_resnet=''
                 ):
        self.data_loader = data_loader
        self.device = device
        self.print_freq_interval = print_freq
        self.is_train_all_size = is_train_all_size
        # 需要修改
        is_diff_mods = False
        self.siren = Siren_Modulation(
            num_inner_layers=9,
            in_channels=2,
            out_channels=3,
            base_channels=256,
            latent_dim=num_modulation,
            is_diff_mods=is_diff_mods,
            is_shift=True,
            is_residual=True,
            bias=True, expansions=[1]
        )

        if is_diff_mods:
            _out_channels = self.siren.modulation_dims
        else:
            _out_channels = num_modulation

        self.para = torch.nn.Parameter(torch.zeros(1, _out_channels))

        self.img_size = img_size
        # self.coordinates = to_coordinates(self.img_size)
        # self.coordinates = self.coordinates.to(device)

        self.num_modulation = num_modulation
        
        self.optimizer_w = torch.optim.AdamW(self.siren.parameters(), lr=1e-5)
        self.optimizer_b = torch.optim.SGD([self.para], lr=1e-2)

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, T_max=500, eta_min=1e-7)
        self.loss_func = torch.nn.MSELoss()
        self.resnet = _resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Identity()
        if pre_train_resnet is not None and os.path.exists(pre_train_resnet):
            print(f'loading {pre_train_resnet}!')
            self.resnet.load_state_dict(torch.load(pre_train_resnet), strict=False)
        self.resnet.eval()

        self.max_epoch = max_epoch
        self.model_dir = model_dir
        self.max_epoch = max_epoch
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.result_dir = result_dir
        if result_dir is None:
            self.result_dir = self.model_dir + '/results'
        os.makedirs(self.result_dir, exist_ok=True)

        self.logger = Logger(self.result_dir + f'/{pattern}_logger.log')

        if os.path.exists(load_checkpoint):
            self.logger.write(f'load checkpoint from {load_checkpoint}')
            states = torch.load(load_checkpoint)
            self.siren.load_state_dict(states['siren'])
        else:
            self.logger.write('train from scratch ......')
        self.siren = self.siren.to(device)
        self.resnet = self.resnet.to(device)

    def train(self):
        for i_epoch in range(self.max_epoch):
            state_dict = {
                    'siren': self.siren.state_dict()
                }
            if i_epoch % 30 == 0:
                torch.save(state_dict, f'{self.model_dir}/{i_epoch}.pth')
            torch.save(state_dict, f'{self.model_dir}/latest.pth')

            for batch_id, data in enumerate(self.data_loader):
                img = data['img']
                img_metas = data['img_meta']
                if self.is_train_all_size:
                    img = [rearrange(x.to(self.device), 'C H W -> (H W) C') for x in img]
                else:
                    img = img.to(self.device)  # B C H W
                    img = rearrange(img, 'B C H W -> B (H W) C')

                modulations_tmp = []
                self.siren.freeze_model_w_b()
                
                for i_batch in range(len(img)):
                    self.para.data = torch.zeros_like(self.para).to(self.device)
                    self.para.requires_grad = True
                    coordinates = to_coordinates(img_metas[i_batch]['img_shape'][:2]).to(self.device)
                    for i_inner_iter in range(5):
                        predicted = self.siren(coordinates, self.para)
                        loss1 = self.loss_func(predicted, img[i_batch])
                        predicted = rearrange(predicted, '(H W) C -> C H W', H=img_metas[i_batch]['img_shape'][1])
                        targets = rearrange(img[i_batch], '(H W) C -> C H W', H=img_metas[i_batch]['img_shape'][1])
                        resnet_input = torch.stack((predicted, targets), dim=0)

                        resnet_input = F.interpolate(resnet_input, size=(28, 28))
                        features = self.resnet(resnet_input)
                        similarity = torch.cosine_similarity(features[0], features[1], dim=0)
                        loss2 = 1 - similarity
                        loss = loss1 + 0.1*loss2
                        self.optimizer_b.zero_grad()
                        loss.backward()
                        self.optimizer_b.step()
                    modulations_tmp.append(self.para.data)

                self.para.requires_grad = False
                self.siren.train_model_w_b()
                losses1 = []
                losses2 = []
                psnres = []
                for i_batch in range(len(img)):
                    modulation = modulations_tmp[i_batch]
                    coordinates = to_coordinates(img_metas[i_batch]['img_shape'][:2]).to(self.device)
                    predicted = self.siren(coordinates, modulation)
                    psnres.append(get_clamped_psnr(predicted, img[i_batch]))

                    loss1 = self.loss_func(predicted, img[i_batch])
                    predicted = rearrange(predicted, '(H W) C -> C H W', H=img_metas[i_batch]['img_shape'][1])
                    targets = rearrange(img[i_batch], '(H W) C -> C H W', H=img_metas[i_batch]['img_shape'][1])
                    resnet_input = torch.stack((predicted, targets), dim=0)

                    resnet_input = F.interpolate(resnet_input, size=(28, 28))
                    features = self.resnet(resnet_input)
                    similarity = torch.cosine_similarity(features[0], features[1], dim=0)
                    loss2 = 1 - similarity

                    losses1.append(loss1)
                    losses2.append(loss2)
                losses = sum(losses1) + 0.1 * sum(losses2)
                self.optimizer_w.zero_grad()
                losses.backward()
                self.optimizer_w.step()
                if batch_id % self.print_freq_interval == 0:
                    self.logger.write(f'{i_epoch}: {batch_id}/{len(self.data_loader)}:  {np.mean(psnres)} loss: {losses.data.cpu().numpy()/len(img)}')
            
            # self.lr_scheduler.step()

    def val(self):
        for batch_id, data in enumerate(self.data_loader):
            img = data['img']
            img_metas = data['img_meta']
            if self.is_train_all_size:
                img = [rearrange(x.to(self.device), 'C H W -> (H W) C') for x in img]
            else:
                img = img.to(self.device)  # B C H W
                img = rearrange(img, 'B C H W -> B (H W) C')
            
            self.siren.freeze_model_w_b()

            for i_batch in range(len(img)):
                self.para.data = torch.zeros_like(self.para).to(self.device)
                self.para.requires_grad = True

                log_dict = img_metas[i_batch]
                log_dict.update(
                    {'modulations': None,
                     'best_psnr': 0,
                     'min_loss': 1e5}
                )

                best_recon_img = None
                coordinates = to_coordinates(img_metas[i_batch]['img_shape'][:2]).to(self.device)
                with tqdm.trange(3000, ncols=100) as t:
                    for i_inner_iter in t:
                        predicted = self.siren(coordinates, self.para)
                        psnr = get_clamped_psnr(predicted, img[i_batch])

                        loss1 = self.loss_func(predicted, img[i_batch])
                        predicted = rearrange(predicted, '(H W) C -> C H W', H=img_metas[i_batch]['img_shape'][1])
                        targets = rearrange(img[i_batch], '(H W) C -> C H W', H=img_metas[i_batch]['img_shape'][1])
                        resnet_input = torch.stack((predicted, targets), dim=0)

                        resnet_input = F.interpolate(resnet_input, size=(28, 28))
                        features = self.resnet(resnet_input)
                        similarity = torch.cosine_similarity(features[0], features[1], dim=0)
                        loss2 = 1 - similarity
                        loss = loss1 + 0.1 * loss2

                        self.optimizer_b.zero_grad()
                        loss.backward()
                        self.optimizer_b.step()

                        self.logger.write_logs = {
                            'loss': loss.item(),
                            'min_loss': log_dict['min_loss'],
                            'psnr': psnr,
                            'best_psnr': log_dict['best_psnr']
                            }
                        t.set_postfix(**self.logger.write_logs)
                        if psnr > log_dict['best_psnr']:
                            log_dict['best_psnr'] = psnr
                        if loss.item() < log_dict['min_loss']:
                            log_dict['min_loss'] = loss.item()
                            log_dict['modulations'] = self.para.data.cpu().numpy()
                            best_recon_img = predicted

                img_meta = img_metas[i_batch]
                file_name = os.path.basename(os.path.dirname(img_meta['file_name'])) + '/' + os.path.basename(img_meta['file_name'])
                os.makedirs(os.path.dirname(self.result_dir +'/' + file_name), exist_ok=True)
                pickle.dump(log_dict, open(self.result_dir +'/'+file_name+f'_{img_meta["img_shape"][0]}.pkl', 'wb'))
                img_recon = best_recon_img.reshape(*img_meta['img_shape'][:2], 3).permute(2, 0, 1).float()
                save_image(torch.clamp(img_recon, 0, 1).cpu(), self.result_dir +'/'+ file_name+f'_{img_meta["img_shape"][0]}.png')
