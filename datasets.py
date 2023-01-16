import torch
from torch.utils.data import DataLoader, Dataset
from skimage import io
from torchvision import transforms
import numpy as np


class INRDataset(Dataset):
    def __init__(
        self, path_prefix, img_file_list, 
        is_load_all=True, img_size=(32, 32), 
        is_test=False, is_train_all_size=False):
        self.img_size = img_size
        self.path_prefix = path_prefix
        self.img_file_list = img_file_list
        self.is_train_all_size = is_train_all_size
        with open(self.img_file_list) as f:
            self.samples = [x.strip().rsplit('.tif', 1) for x in f.readlines()]
        self.samples = [self.path_prefix + '/' + x[0] + '.tif' for x in self.samples]
        self.is_load_all = is_load_all
        if self.is_load_all:
            self.all_imgs = []
            for idx, i_samples in enumerate(self.samples):
                self.all_imgs.append(io.imread(i_samples, as_gray=False))
        self.is_test = is_test

    def __len__(self):
        return len(self.samples)
    
    def get_train_data(self, img):
        img_meta = {}
        h, w, c = img.shape
        img = transforms.ToTensor()(img)
        # img = transforms.RandomResizedCrop(self.img_size, scale=(self.img_size[0] / w - 0.05, 1))(img)
        if self.is_train_all_size == 1:
            sizes = [(28, 28), (56, 56), (112, 112)]
            rand_idx = torch.randint(len(sizes), size=[])
            img_size = sizes[rand_idx]
            img = transforms.Resize(size=img_size)(img)
        elif self.is_train_all_size == 2:
            sizes = {56: (28, 28), 112:(56, 56), 224:(112, 112)}
            img_size = sizes[h]
            img = transforms.Resize(size=img_size)(img)
            if h == 224:
                img = transforms.CenterCrop(size=(84, 84))(img)   
        elif h > self.img_size[0]:
            # img = transforms.CenterCrop(self.img_size)(img)
            img = transforms.Resize(size=self.img_size)(img)
            # if torch.randn([]) > 0.5:
            #     img = transforms.Resize(size=self.img_size)(img)
            # else:
            #     img = transforms.CenterCrop(self.img_size)(img)
        # img = transforms.RandomHorizontalFlip(p=0.3)(img)
        # img = transforms.RandomVerticalFlip(p=0.3)(img)
        img_meta['is_train_all_size'] = self.is_train_all_size
        img_meta['img_shape'] = (*img.size()[1:], 3)
        return img, img_meta
    
    def get_test_data(self, img):
        img_meta = {}
        h, w, c = img.shape
        img_meta['ori_img_shape'] = (h, w, c)
        img = transforms.ToTensor()(img)

        if self.is_train_all_size == 1:
            sizes = [(28, 28), (56, 56), (112, 112)]
            rand_idx = torch.randint(len(sizes), size=[])
            img_size = sizes[rand_idx]
            img = transforms.Resize(size=img_size)(img)
        elif self.is_train_all_size == 2:
            sizes = {56: (28, 28), 112:(56, 56), 224:(112, 112)}
            img_size = sizes[h]
            img = transforms.Resize(size=img_size)(img)
            if h == 224:
                img = transforms.CenterCrop(size=(84, 84))(img)   
        elif self.is_train_all_size == 0:
            if h > self.img_size[0]:
                if h == 224:
                    img = transforms.Resize(size=(112, 112))(img)
                    img = transforms.CenterCrop(size=(84, 84))(img)
                img = transforms.Resize(size=self.img_size)(img)
               
        
        img_meta['img_shape'] = (*img.size()[1:], 3)
        img_meta['is_train_all_size'] = self.is_train_all_size
        return img, img_meta

    def __getitem__(self, item):
        if hasattr(self, 'all_imgs'):
            img = self.all_imgs[item]
        else:
            img = io.imread(self.samples[item], as_gray=False)
        
        if self.is_test:
            img, img_meta = self.get_test_data(img)
            img_meta['file_name'] = self.samples[item]
        else:
            img, img_meta = self.get_train_data(img)
        
        return {'img': img, 'img_meta': img_meta}


def get_dataloader(dataset, batch_size=8, shuffle=False, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=True, collate_fn=collate_fn)


def collate_fn(batch):
    def to_tensor(item):
        if torch.is_tensor(item):
            return item
        elif isinstance(item, type(np.array(0))):
            return torch.from_numpy(item).float()
        elif isinstance(item, type('0')):
            return item
        elif isinstance(item, list):
            return item
        elif isinstance(item, dict):
            return item
        else:
            return item

    return_data = {}
    for key in batch[0].keys():
        return_data[key] = []

    for sample in batch:
        for key, value in sample.items():
            return_data[key].append(to_tensor(value))

    if return_data['img_meta'][0].get('is_train_all_size', 0) != 0 :
        return return_data
    
    keys = ['img']
    for key in keys:
        return_data[key] = torch.stack(return_data[key], dim=0)

    return return_data
