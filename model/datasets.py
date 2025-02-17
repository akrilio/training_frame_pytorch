import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# import torchvision.transforms.functional as F


class BasicDataset(Dataset):
    def __init__(self, data, input_shape, random=None):
        self.data = data
        self.input_shape = input_shape
        self.normal_param = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
        self.transform = self.create_transform(random)

    def create_transform(self, augment_type='strong'):
        if augment_type == 'weak':
            return transforms.Compose([
                transforms.Resize(self.input_shape),
                transforms.RandomCrop(self.input_shape, padding=max(self.input_shape) // 8, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.normal_param['mean'], self.normal_param['std'])
            ])
        elif augment_type == 'medium':
            return transforms.Compose([
                transforms.Resize(self.input_shape),
                transforms.RandomCrop(self.input_shape, padding=max(self.input_shape) // 8, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=1, magnitude=5),
                transforms.ToTensor(),
                transforms.Normalize(self.normal_param['mean'], self.normal_param['std'])
            ])
        elif augment_type == 'strong':
            return transforms.Compose([
                # transforms.Resize(self.input_shape),
                # transforms.RandomCrop(self.input_shape, padding=max(self.input_shape) // 8, padding_mode='constant'),
                transforms.RandomResizedCrop(self.input_shape, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=(0.4, 2.), hue=0.05),
                transforms.RandAugment(num_ops=3, magnitude=5),
                transforms.ToTensor(),
                # transforms.RandomErasing(scale=(0.02, 0.2), value=0),
                transforms.Normalize(self.normal_param['mean'], self.normal_param['std']),
            ])
        elif augment_type == 'valid' or augment_type is None:
            return transforms.Compose([
                transforms.Resize(self.input_shape),
                transforms.ToTensor(),
                transforms.Normalize(self.normal_param['mean'], self.normal_param['std']),
            ])

    def __getitem__(self, i):
        image = Image.open(self.data['image'][i])
        label = self.data['label'][i]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_data = self.transform(image).to(torch.float32, memory_format=torch.contiguous_format)
        label_data = torch.tensor(label).to(torch.int64, memory_format=torch.contiguous_format)
        return image_data, label_data

    def __len__(self):
        return len(self.data['image'])


class CifarDataset(BasicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        image, label = self.data['image'][i], self.data['label'][i]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_data = self.transform(image)
        label_data = torch.tensor(label)
        return image_data, label_data


class SSLDataset(BasicDataset):
    def __init__(self, data, input_shape, random=None):
        super().__init__(data, input_shape, None)
        self.transform = {f"{key}_{n}": self.create_transform(key) for n, key in enumerate(random)}

    def __getitem__(self, i):
        image = Image.open(self.data['image'][i])
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_data = [trans(image).to(torch.float32) for trans in self.transform.values()]
        return image_data
