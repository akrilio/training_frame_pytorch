import os
import glob
import random
import json
import numpy as np

random.seed(None)


#  [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


class DatasetEditor(object):
    def __init__(self, dataset_path=None, split=(6, 2, 2)):
        # ori_path/*classes/(image & label)/*idx.png
        self.dataset_path = dataset_path
        if os.path.exists(dataset_path) is False:
            os.mkdir(dataset_path)
        self.data = {"train": {}, "valid": {}, "test": {}}
        self.split = np.cumsum(split, axis=0)

    def load_cifar10(self, load_path=r'datasets/cifar10'):
        import torchvision as tv
        if os.path.exists(load_path) is False:
            os.mkdir(load_path)
        cifar10 = tv.datasets.CIFAR10(root=load_path, train=True, download=True)
        cifar10_test = tv.datasets.CIFAR10(root=load_path, train=False, download=True)
        return cifar10, cifar10_test

    def save_cifar10_label_idx(self):
        train_data = self.data['train']
        label_idx = {}
        for i in range(10):
            idx = np.where(np.array(train_data['label']) == i)[0]
            label_idx[str(i)] = idx.tolist()
        with open(os.path.join(self.dataset_path, "label_index.json"), 'w') as write_f:
            write_f.write(json.dumps(label_idx, ensure_ascii=False))
        write_f.close()

    def read_cifar(self, data_path=r'datasets/cifar10'):
        import pickle
        search = os.path.join(data_path, "data_batch_*")
        train_data = glob.glob(search)
        for key in self.data.keys():
            self.data[key]['label'] = []
            self.data[key]['image'] = []
            self.data[key]['filename'] = []
        for f in train_data:
            with open(f, 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
            fo.close()
            self.data['train']['label'] += dic[b'labels']
            data = np.array(dic[b'data'].reshape(-1, 3, 32, 32)).transpose(0, 2, 3, 1)
            self.data['train']['image'].append(data)
            self.data['train']['filename'] += dic[b'filenames']
        self.data['train']['image'] = np.concatenate(self.data['train']['image'], axis=0)
        search = os.path.join(data_path, "test_batch")
        test_data = glob.glob(search)
        for f in test_data:
            with open(f, 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
            fo.close()
            self.data['test']['label'] += dic[b'labels']
            data = np.array(dic[b'data'].reshape(-1, 3, 32, 32)).transpose(0, 2, 3, 1)
            self.data['test']['image'].append(data)
            self.data['test']['filename'] += dic[b'filenames']
            self.data['valid']['label'] += dic[b'labels']
            data = np.array(dic[b'data'].reshape(-1, 3, 32, 32)).transpose(0, 2, 3, 1)
            self.data['valid']['image'].append(data)
            self.data['valid']['filename'] += dic[b'filenames']
        self.data['test']['image'] = np.concatenate(self.data['test']['image'], axis=0)
        self.data['valid']['image'] = np.concatenate(self.data['valid']['image'], axis=0)
        with open(os.path.join(data_path, 'batches.meta'), 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
            self.data['label_names'] = dic[b'label_names']
        fo.close()
        with open(os.path.join(data_path, 'label_index.json'), 'r') as data_file:
            self.data['label_idx'] = json.load(data_file)
        data_file.close()
        return self.data

    def read_cityscapes(self, data_path=r'datasets/cityscapes_resized', r=1):
        for key in self.data.keys():
            path = os.path.join(data_path, '{}.json'.format(key))
            with open(path, 'r') as data_file:
                self.data[key] = r * list(json.load(data_file))
            data_file.close()
        from datasets.cityscapes.labels import labels
        id = {label.trainId: label.name for label in labels}
        id[19] = 'void'
        id.pop(255), id.pop(-1)
        return self.data, id


if __name__ == '__main__':
    editor = Dataset_Editor(dataset_path=r'datasets\cifar10')
    data = editor.read_cifar()
    for key, value in data['label_idx'].items():
        print(key, len(value))
    """
    from model.datasets import BasicDataset, SSLDataset, Mixer
    import matplotlib.pyplot as plt
    import torch

    # dataset = BasicDataset(data['train'], input_shape=(32, 32), num_classes=10, random='strong')
    dataset = SSLDataset(data['train'], input_shape=(32, 32), random=['weak', 'strong'])
    fig, ax = plt.subplots(2, 4)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # mix_up = Mixer(input_shape=(64, 3, 32, 32), unpack_fn=lambda data: [torch.cat(d, dim=0) for d in data])
    class_idx = torch.arange(10).reshape(1, -1)
    for n, (image_data, label_data) in enumerate(data_loader, 0):
        if n >= 4:
            break
        image = [image_data.permute(0, 2, 3, 1).cpu().detach().numpy(),
                 label_data.permute(0, 2, 3, 1).cpu().detach().numpy()]
        # label = torch.sum(class_idx * label_data, dim=1).cpu().detach().numpy()
        for i in range(2):
            ax[i][n].imshow(image[i].reshape(-1, 32, 3))
            # ax[i][n].set_title('label:{}'.format(data['label_names'][label[i]]))
    plt.show()
    """