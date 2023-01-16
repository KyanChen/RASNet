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


class Trainer():
    def __init__(self, representation, batch_size=256, max_epoch=500000, lr=1e-3, print_freq=5, device='cuda'):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.device = device
        self.representation = representation
        self.modulation_size_dict = self.representation.get_bias_size()
        _out_channels = sum(self.modulation_size_dict.values())

        self.para = torch.nn.Parameter(torch.zeros(1, 512))
        self.img_size = (32, 32)
        self.coordinates = to_coordinates(self.img_size)
        self.coordinates = self.coordinates.to(device)

        # para_w = []
        # para_b = []
        # for name, parm in self.representation.named_parameters():
        #     if 'weight' in name:
        #         para_w.append(parm)
        #     elif 'bias' in name:
        #         para_b.append(parm)
        #     else:
        #         raise "error"
        self.optimizer_w = torch.optim.AdamW(self.representation.parameters(), lr=1e-5)
        self.optimizer_b = torch.optim.SGD([self.para], lr=1e-2)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())
        # self.path_prefix = r'D:\GID\RGB_15_train'
        self.path_prefix = 'RGB_15_train'
        self.img_file_list = self.path_prefix + '/../all_img_list_56.txt'
        with open(self.img_file_list) as f:
            self.samples = [x.strip().rsplit('.tif', 1) for x in f.readlines()]
        self.samples = [self.path_prefix+'/'+x[0]+'.tif' for x in self.samples]
        self.ori_samples = self.samples
        self.batch_size = batch_size
        self.total_size = int(np.ceil(len(self.samples) / batch_size) * batch_size)
        self.samples = np.array((2*self.samples)[:self.total_size])
        self.max_epoch = max_epoch
        self.is_load_all = False
        if self.is_load_all:
            self.all_imgs = torch.zeros(len(self.samples), np.prod(self.img_size), 3).float()
            for idx, i_samples in enumerate(self.samples):
                img = io.imread(i_samples, as_gray=False)
                img = transforms.ToTensor()(img)
                img = transforms.Resize(self.img_size)(img).float()
                img = rearrange(img, 'c h w -> (h w) c')
                self.features[idx] = img
        # states = torch.load('models/latest.pth')
        # self.representation.load_state_dict(states)

    def train(self):
        os.makedirs('models', exist_ok=True)
        for i_epoch in range(self.max_epoch):
            inds = torch.randperm(self.total_size)
            if i_epoch % 10 == 0:
                torch.save(self.representation.state_dict(), f'models/{i_epoch}.pth')
                torch.save(self.representation.state_dict(), f'models/latest.pth')
                pass

            if not self.is_load_all:
                self.features = torch.zeros(len(self.samples), np.prod(self.img_size), 3).float()
                for idx, i_samples in enumerate(self.samples):
                    img = io.imread(i_samples, as_gray=False)
                    img = transforms.ToTensor()(img)
                    img = transforms.RandomResizedCrop(self.img_size, scale=(0.4, 0.8))(img)
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    img = rearrange(img, 'c h w -> (h w) c')
                    self.features[idx] = img

            for i_out_iter in range(self.total_size // self.batch_size):
                sample_inds = inds[i_out_iter*self.batch_size:(i_out_iter+1)*self.batch_size].cpu().numpy()
                feature = self.features[sample_inds]
                feature = feature.to(self.device)
                modulations_tmp = []
                self.representation.freeze_model_w_b()
                self.para.requires_grad = True
                for batch_id in range(self.batch_size):
                    self.para.data = torch.zeros_like(self.para)
                    for i_inner_iter in range(6):
                        predicted = self.representation(self.coordinates, self.para)
                        loss = self.loss_func(predicted, feature[batch_id])
                        self.optimizer_b.zero_grad()
                        loss.backward()
                        self.optimizer_b.step()
                        # psnr = get_clamped_psnr(predicted, feature[batch_id])
                        # print(i_inner_iter, ": ", psnr)
                    modulations_tmp.append(self.para.data)
                    # modulations_tmp.append(self.representation.get_model_b_data())
                    if batch_id == 1:
                        psnr = get_clamped_psnr(predicted, feature[batch_id])
                        print(batch_id, ": ", psnr)

                self.para.requires_grad = False
                self.representation.train_model_w_b()
                losses = []
                psnres = []
                # self.para.requires_grad = True
                for batch_id in range(self.batch_size):
                    modulation = modulations_tmp[batch_id]
                    # self.representation.set_model_b_data(modulations_tmp[batch_id])
                    predicted = self.representation(self.coordinates, modulation)
                    loss = self.loss_func(predicted, feature[batch_id])
                    psnres.append(get_clamped_psnr(predicted, feature[batch_id]))
                    losses.append(loss)
                losses = sum(losses)
                self.optimizer_w.zero_grad()
                losses.backward()
                self.optimizer_w.step()
                print("all batch psnr: ", np.mean(psnres))

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


