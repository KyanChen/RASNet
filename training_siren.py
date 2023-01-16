import pickle
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
from siren_modulation_v1 import Siren_Modulation as Siren_Modulation_V1
from logger import Logger


class Trainer:
    def __init__(self,
                 data_loader, img_size=(32, 32),
                 num_modulation=1024, max_epoch=50000, print_freq=5, device='cuda',
                 model_dir='tmp', load_checkpoint='', result_dir=None, vis_metric=True,
                 is_train_all_size=False,
                 pattern='train', is_BN=False,
                 ):
        self.data_loader = data_loader
        self.device = device
        self.print_freq_interval = print_freq
        self.vis_metric = vis_metric
        self.is_train_all_size = is_train_all_size
        self.is_BN =is_BN
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
            is_residual=False,
            is_BN=is_BN,
            bias=True, expansions=[1]
            )

        if is_diff_mods:
            _out_channels = self.siren.modulation_dims
        else:
            _out_channels = num_modulation
        self._out_channels = _out_channels
        self.para = torch.nn.Parameter(torch.zeros(1, _out_channels))

        self.img_size = img_size
        # self.coordinates = to_coordinates(self.img_size)
        # self.coordinates = self.coordinates.to(device)

        self.num_modulation = num_modulation
        
        self.optimizer_w = torch.optim.AdamW(self.siren.parameters(), lr=1e-5)
        self.optimizer_b = torch.optim.SGD([self.para], lr=1e-2)

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, T_max=500, eta_min=1e-7)
        self.loss_func = torch.nn.MSELoss()

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
            self.siren.load_state_dict(states['siren'], strict=False)
        else:
            self.logger.write('train from scratch ......')
        self.siren = self.siren.to(device)

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
                psnres = []
                for i_batch in range(len(img)):
                    self.para.data = torch.zeros_like(self.para).to(self.device)
                    self.para.requires_grad = True
                    coordinates = to_coordinates(img_metas[i_batch]['img_shape'][:2]).to(self.device)
                    for i_inner_iter in range(5):
                        predicted = self.siren(coordinates, self.para)
                        loss = self.loss_func(predicted, img[i_batch])
                        self.optimizer_b.zero_grad()
                        loss.backward()
                        self.optimizer_b.step()
                        # if i_batch == 0:
                        #     psnr = get_clamped_psnr(predicted, img[i_batch])
                        #     self.logger.write(f'{i_inner_iter}:  {psnr}')
                    modulations_tmp.append(self.para.data)
                    psnres.append(get_clamped_psnr(predicted, img[i_batch]))
                if self.vis_metric:
                    self.logger.write(f'vis_metric {i_epoch}: {batch_id}/{len(self.data_loader)}:  {np.mean(psnres)}')

                self.para.requires_grad = False
                self.siren.train_model_w_b()

                # coordinate_batch = []
                # targets_batch = []
                # modulations_batch = []
                # for i_batch in range(len(img)):
                #     h, w = img_metas[i_batch]['img_shape'][:2]
                #     coordinates = to_coordinates((h, w)).to(self.device)  # Nx2
                #     targets = img[i_batch]  # Nx3
                #     modulation = modulations_tmp[i_batch]
                #     modulations = repeat(modulation, '1 n_dims -> N n_dims', N=len(targets))
                #     coordinate_batch.append(coordinates)
                #     targets_batch.append(targets)
                #     modulations_batch.append(modulations)
                # coordinate_batch = torch.cat(coordinate_batch)
                # targets_batch = torch.cat(targets_batch)
                # modulations_batch = torch.cat(modulations_batch)
                # predicted = self.siren(coordinate_batch, modulations_batch)
                # loss = self.loss_func(predicted, targets_batch)
                # psnres = get_clamped_psnr(predicted.data, targets_batch.data)
                # losses = loss

                losses = []
                psnres = []
                for i_batch in range(len(img)):
                    modulation = modulations_tmp[i_batch]
                    coordinates = to_coordinates(img_metas[i_batch]['img_shape'][:2]).to(self.device)
                    predicted = self.siren(coordinates, modulation)
                    loss = self.loss_func(predicted, img[i_batch])
                    psnres.append(get_clamped_psnr(predicted, img[i_batch]))
                    losses.append(loss)
                losses = sum(losses) / len(img)

                self.optimizer_w.zero_grad()
                losses.backward()
                self.optimizer_w.step()
                if not self.vis_metric:
                    if batch_id % self.print_freq_interval == 0:
                        self.logger.write(f'{i_epoch}: {batch_id}/{len(self.data_loader)}:  {np.mean(psnres)} loss: {losses.data.cpu().numpy()}')
            
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
                        loss = self.loss_func(predicted, img[i_batch])
                        psnr = get_clamped_psnr(predicted, img[i_batch])
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
                            log_dict['modulations'] = self.para.data
                            best_recon_img = predicted
                        if loss.item() < log_dict['min_loss']:
                            log_dict['min_loss'] = loss.item()

                        # img_meta = img_metas[i_batch]
                        # file_name = os.path.basename(os.path.dirname(img_meta['file_name'])) + '/' + os.path.basename(
                        #     img_meta['file_name'])
                        # os.makedirs(os.path.dirname(self.result_dir + '/' + file_name), exist_ok=True)
                        # pickle.dump(log_dict, open(
                        #     self.result_dir + '/' + file_name + f'_{img_meta["is_resize"]}_{img_meta["is_center_crop"]}.pkl',
                        #     'wb'))
                        # img_recon = best_recon_img.reshape(h, w, 3).permute(2, 0, 1).float()
                        # save_image(torch.clamp(img_recon, 0, 1).cpu(),
                        #            self.result_dir + '/' + file_name + f'_{img_meta["is_resize"]}_{img_meta["is_center_crop"]}_{i_inner_iter}.png')
                if self.is_BN:
                    log_dict['modulations'] = self.siren.get_BN_feature(log_dict['modulations']).data
                log_dict['modulations'] = log_dict['modulations'].cpu().numpy()

                img_meta = img_metas[i_batch]
                file_name = os.path.basename(os.path.dirname(img_meta['file_name'])) + '/' + os.path.basename(img_meta['file_name'])
                os.makedirs(os.path.dirname(self.result_dir +'/' + file_name), exist_ok=True)
                pickle.dump(log_dict, open(self.result_dir +'/'+file_name+f'_{img_meta["img_shape"][0]}.pkl', 'wb'))
                img_recon = best_recon_img.reshape(*img_meta['img_shape'][:2], 3).permute(2, 0, 1).float()
                save_image(torch.clamp(img_recon, 0, 1).cpu(), self.result_dir +'/'+ file_name+f'_{img_meta["img_shape"][0]}.png')

    def val_batch(self):
        for batch_id, data in enumerate(self.data_loader):
            img = data['img']
            img_metas = data['img_meta']
            if self.is_train_all_size:
                img = [rearrange(x.to(self.device), 'C H W -> (H W) C') for x in img]

            else:
                img = img.to(self.device)  # B C H W
                img = rearrange(img, 'B C H W -> B (H W) C')
            
            self.siren.freeze_model_w_b()
            modulations_tmp = [torch.nn.Parameter(torch.zeros(1, self._out_channels)) for _ in range(len(img))]
            optimizer_b = torch.optim.SGD(modulations_tmp, lr=1e-2)

            coordinate_batch = []
            targets_batch = []
            modulations_batch = []
            for i_batch in range(len(img)):
                h, w = img_metas[i_batch]['img_shape'][:2]
                coordinates = to_coordinates((h, w))  # Nx2
                targets = img[i_batch]  # Nx3
                modulation = modulations_tmp[i_batch]
                # modulation.data = modulation.data.to(self.device)
                modulations = modulation.expand((len(targets), modulation.size(1)))
                # modulations = repeat(modulation, '1 n_dims -> N n_dims', N=len(targets))
                coordinate_batch.append(coordinates)
                targets_batch.append(targets)
                modulations_batch.append(modulations)

            coordinate_batch = torch.cat(coordinate_batch).to(self.device)
            targets_batch = torch.cat(targets_batch)
            modulations_batch = torch.cat(modulations_tmp)
            modulations_batch.data = modulations_batch.data.to(self.device)

            log_dict = img_metas[0]
            log_dict.update(
                    {
                     'best_psnr': 0,
                     'min_loss': 1e5}
                )
            best_recon_imgs = None
            best_paras = None
            with tqdm.trange(1000, ncols=100) as t:
                for i_inner_iter in t:
                    predicted = self.siren(coordinate_batch, modulations_batch)
                    loss = self.loss_func(predicted, targets_batch)
                    optimizer_b.zero_grad()
                    loss.backward()
                    optimizer_b.step()
                    psnres = get_clamped_psnr(predicted.data, targets_batch.data)
                    psnr = np.mean(psnres)
                    self.logger.write_logs = {
                        'loss': loss.item(),
                        'min_loss': log_dict['min_loss'],
                        'psnr': psnr,
                        'best_psnr': log_dict['best_psnr']
                        }
                    t.set_postfix(**self.logger.write_logs)
                    if psnr > log_dict['best_psnr']:
                        log_dict['best_psnr'] = psnr
                        best_paras = [x.data for x in modulations_tmp]
                        best_recon_imgs = predicted
                    if loss.item() < log_dict['min_loss']:
                        log_dict['min_loss'] = loss.item()
            
            if self.is_BN:
                best_paras = torch.cat(best_paras, 0)
                log_dict['modulations'] = self.siren.get_BN_feature(best_paras).data

            log_dict['modulations'] = log_dict['modulations'].cpu().numpy()
            
            id_pixels = 0
            for i_batch in range(len(img)):
                img_meta = img_metas[i_batch]
                log_dict = img_meta
                log_dict.update(
                    {'modulations': best_paras[i_batch]}
                )
                h, w = img_metas[i_batch]['img_shape'][:2]
                best_recon_img = best_recon_imgs[id_pixels:id_pixels+(h*w)]
                id_pixels += h*w
                file_name = os.path.basename(os.path.dirname(img_meta['file_name'])) + '/' + os.path.basename(img_meta['file_name'])
                os.makedirs(os.path.dirname(self.result_dir +'/' + file_name), exist_ok=True)
                pickle.dump(log_dict, open(self.result_dir +'/'+file_name+f'_{img_meta["img_shape"][0]}_{img_meta["is_resize"]}.pkl', 'wb'))
                img_recon = best_recon_img.reshape(*img_meta['img_shape'][:2], 3).permute(2, 0, 1).float()
                save_image(torch.clamp(img_recon, 0, 1).cpu(), self.result_dir +'/'+ file_name+f'_{img_meta["img_shape"][0]}_{img_meta["is_resize"]}.png')
