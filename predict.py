import json
import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from model.resnet import ResNetAtt
from model.datasets import BasicDataset, SSLDataset
from dataset_edit import Dataset_Editor
from image_searcher import Searcher


class Predictor(object):
    def __init__(self, model, dataset_path, param_path, image_shape=(32, 32), key='test'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = image_shape
        if dataset_path is not None:
            self.dataset, self.classes = self.read_dataset(dataset_path, key)
            self.num_classes = len(self.classes)
            self.matrix = np.zeros(shape=(self.num_classes, self.num_classes))

        if param_path is not None:
            self.model = self.load_model(model, param_path)
            self.param_path = param_path

        if dataset_path and param_path:
            assert self.model.params['num_classes'] == self.num_classes

    def read_dataset(self, dataset_path, key='valid'):
        editor = Dataset_Editor(dataset_path)
        data = editor.read_cifar()
        dataset = BasicDataset(
            data[key], input_shape=self.image_shape, num_classes=len(data['label_names']), random='valid'
        )
        return dataset, data['label_names']

    def load_model(self, model_type, param_path):
        with open(os.path.join(param_path, 'model_params.json'), 'r') as data_file:
            param = json.load(data_file)
        data_file.close()
        # weights_path = os.path.join(param_path, 'checkpoint.pt')
        weights_path = os.path.join(param_path, 'last_epoch_*.pt')
        weights_path = glob.glob(weights_path)[-1]
        print('load weights from:{}'.format(weights_path))
        model = model_type(**param)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.to(self.device)
        return model

    def show_results(self, image, label, pre=None):
        results = 'label: {}'.format(self.classes[label])
        if pre is not None:
            results += 'pre: {}'.format(self.classes[pre])
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(results)
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tight_layout()
        plt.show()

    def __call__(self, image, label, show=False):
        shape = image.shape
        image_data = torch.permute(torch.tensor(image).reshape((1,) + shape),
                                   (0, 3, 1, 2)).to(self.device).to(torch.float32)
        self.model.eval()
        pre = self.model(image_data)
        pre = torch.argmax(pre, dim=1).squeeze(0).cpu().detach().numpy()
        if show:
            self.show_results(image, label, pre)
        return pre

    def evaluate(self, label, pre):
        label = F.one_hot(label, self.num_classes).reshape(-1, self.num_classes, 1).to(torch.float32)
        pre = F.one_hot(torch.argmax(pre, dim=1), self.num_classes).reshape(-1, 1, self.num_classes).to(torch.float32)
        self.matrix += torch.sum(label @ pre, dim=0).cpu().detach().numpy()

    def add_right_cax(self, ax, pad, width):
        ax_pos = ax.get_position()
        cax_pos = mpl.transforms.Bbox.from_extents(
            ax_pos.x1 + pad,
            ax_pos.y0,
            ax_pos.x1 + pad + width,
            ax_pos.y1
        )
        cax = ax.figure.add_axes(cax_pos)
        return cax

    def show_matrix(self, save=False, path=None):
        dpi = [100, 200][save]
        proportion = self.matrix / np.sum(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix.astype(np.uint)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
        im = ax.imshow(proportion, interpolation='nearest', cmap='Blues')
        # ax.set_title('confusion matrix')
        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks, self.classes, fontsize=10.5)
        ax.set_yticks(tick_marks, self.classes, fontsize=10.5)
        cax = self.add_right_cax(ax, pad=0.02, width=0.02)
        fig.colorbar(im, cax=cax)

        i, j = np.meshgrid(np.arange(self.num_classes), np.arange(self.num_classes))
        iters = np.reshape(np.transpose(np.array([i, j]), (1, 2, 0)), (-1, 2))
        text_proportion = np.vectorize("{:.2f}%".format)(proportion * 100)
        for i, j in iters:
            if i == j:
                ax.text(j, i - 0.12, str(self.matrix[i, j]), va='center', ha='center', fontsize=10.5, color='white')
                ax.text(j, i + 0.12, text_proportion[i, j], va='center', ha='center', fontsize=10.5, color='white')
            else:
                ax.text(j, i - 0.12, str(self.matrix[i, j]), va='center', ha='center', fontsize=10.5)
                ax.text(j, i + 0.12, text_proportion[i, j], va='center', ha='center', fontsize=10.5)

        ax.set_ylabel('true label', fontsize=12)
        ax.set_xlabel('predict label', fontsize=12)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        if save:
            if path is not None:
                plt.savefig(path)
            else:
                plt.savefig(os.path.join(self.param_path, 'confusion_matrix.png'))
        else:
            plt.show()

    def test_on_dataset(self):
        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=32, shuffle=False, pin_memory=True,
            num_workers=4, persistent_workers=True, prefetch_factor=2,
        )
        self.model.eval()
        with torch.no_grad():
            for (image_data, label_data) in data_loader:
                image_data = image_data.to(self.device)
                label_data = label_data.to(self.device)
                pre = self.model(image_data)
                self.evaluate(label_data, pre)


class Distributor(Predictor):
    def __init__(self, distribute_root, integrate_root,
                 model=ResNetAtt, model_path=None, image_shape=(32, 32), key='缺陷',
                 members=['LXQ', 'CJ', 'JYZ', 'LBY', 'CJQ', 'ZY', 'PRZ', 'HYH', 'LZY']):
        self.distribute_root = distribute_root
        self.unlabeled_filename = os.path.join(integrate_root, 'unlabeled_data.json')
        self.manual_filename = os.path.join(integrate_root, 'manual_labeled_data.json')
        super().__init__(model=model, param_path=model_path,
                         dataset_path=self.unlabeled_filename, image_shape=image_shape, key=key)
        self.members = members
        self.max_prob = None
        self.pre_label = None
        self.key = key
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=64, shuffle=False, pin_memory=True,
            num_workers=16, persistent_workers=True, prefetch_factor=2
        )

    def read_dataset(self, dataset_path, key='缺陷'):
        searcher = Searcher(root_path=r'F:\供水管道', dataset_path=r'D:\Public\water_pipeline')
        lb_dataset, ulb_dataset = searcher.read_dataset(label=key, equal=False)
        # dataset = SSLDataset(ulb_dataset, input_shape=self.image_shape, random=['valid'])
        dataset = BasicDataset(lb_dataset['valid'], input_shape=self.image_shape, random='valid')
        if key == 'all':
            classes = ([lb['formal_name'] for lb in lb_dataset['label_names']['缺陷'].values()] +
                       [lb['formal_name'] for lb in lb_dataset['label_names']['管道附属设备及管配件'].values()])
        else:
            classes = [lb['formal_name'] for lb in lb_dataset['label_names'].values()]
        return dataset, classes

    def prob_predict(self):
        self.max_prob = []
        self.pre_label = []
        with torch.no_grad():
            for x in self.data_loader:
                x = x[0].to(self.device)
                y_pre = self.model(x)
                prob = torch.softmax(y_pre, dim=1)
                label = torch.argmax(prob, dim=1)
                max_prob = torch.max(prob, dim=1)[0]
                self.max_prob.append(max_prob)
                self.pre_label.append(label)
        self.max_prob = torch.cat(self.max_prob, dim=0)
        self.pre_label = torch.cat(self.pre_label, dim=0)
        count = 1. * (self.max_prob > 0.95)
        print("Total sample: {}, credible sample: {}, credible ratio: {}"
              .format(len(self.max_prob), torch.sum(count), torch.mean(count)))

    def distribute(self, n=50):
        """
        with open(self.unlabeled_filename, 'r') as file:
            unlabeled_dic = json.load(file)
        file.close()
        """
        num = len(self.members) * n
        index = torch.argsort(self.max_prob, descending=False)
        idx = torch.randperm(num)
        index = index[idx].reshape(-1, n).cpu().numpy()
        for member, idx in zip(self.members, index):
            path = os.path.join(self.distribute_root, member)
            if os.path.exists(path) is False:
                os.makedirs(path)
            for i in idx:
                # image_name = unlabeled_dic[str(i)]['image']
                image_name = self.dataset.data['image'][i]
                image = Image.open(image_name)
                basename = image_name.split('\\')[-1]
                image.save(os.path.join(path, f"ID_{str(i)}_{basename}"))

    def integrate_credible_label(self):
        with open(self.unlabeled_filename, 'r') as file:
            dic = json.load(file)
        file.close()
        for i, (prob, label) in enumerate(zip(self.max_prob.cpu().numpy(), self.pre_label.cpu().numpy())):
            id = self.dataset.data['id'][i]
            dic[id]['pseudo_label'][self.key].append(int(label))
            dic[id]['prob'][self.key].append(float(prob))
            """
            if prob > 0.95:
            else:
                dic[id]['pseudo_label'][self.key].append(-1)
                dic[id]['prob'][self.key].append(0)
            """
        with open(self.unlabeled_filename, 'w') as file:
            file.write(json.dumps(dic, indent=4, ensure_ascii=False))
        file.close()

    def read_manual_label(self, label_name):
        # b = {'t': True, 'f': False}
        with open(label_name, 'r', encoding='utf-8', errors='ignore') as f:
            """
            text = f.readlines()[0: 17]
            image_path = text[-1].split('"')[3]
            flags = {
                t.split('"')[1]:
                b[t.split('"')[-1].replace(': ', '')[0]]
                for t in text[3: 14]
            }
            manual_label = {'imagePath': image_path, 'flags': flags}
            """
            manual_label = json.load(f)
        f.close()
        id = manual_label['imagePath'].split('_')[1]
        flags = np.array(list(manual_label['flags'].keys()))
        masks = list(manual_label['flags'].values())
        selected_names = flags[masks]
        label = [0, 0]
        for name in selected_names:
            if name[:2] == '-1':
                label = [-1, -1]
            else:
                label[int(name[0])] = int(name[1])
        label = {'缺陷': label[0], '管道附属设备及管配件': label[1]}
        return id, label

    def integrate_manual_label(self):
        search = os.path.join(self.distribute_root, '*', '*.json')
        label_names = glob.glob(search)

        if os.path.exists(self.manual_filename):
            with open(self.manual_filename, 'r') as file:
                manual_dic = json.load(file)
            file.close()
        else:
            manual_dic = {}
        with open(self.unlabeled_filename, 'r') as file:
            unlabeled_dic = json.load(file)
        file.close()
        n = 0
        for label_name in label_names:
            id, label = self.read_manual_label(label_name)
            if label['缺陷'] != -1 and id not in manual_dic.keys():
                manual_dic[id] = {'image_name': unlabeled_dic[id]['image_name'],
                                  'pseudo_label': unlabeled_dic[id]['pseudo_label'],
                                  'label': label}
                unlabeled_dic.pop(id)
                n += 1
        """"""
        with open(self.unlabeled_filename, 'w') as file:
            file.write(json.dumps(unlabeled_dic, indent=4, ensure_ascii=False))
        file.close()
        with open(self.manual_filename, 'w', encoding='gbk') as file:
            file.write(json.dumps(manual_dic, indent=4, ensure_ascii=False))
        file.close()
        print("update manual data: {}".format(n))

    def clean_root(self):
        search = os.path.join(self.distribute_root, '*', '*')
        files = glob.glob(search)
        for file in files:
            os.remove(file)


def meaningful_t():
    integrate_root = r'D:\Public\water_pipeline'
    distribute_root = os.path.join(integrate_root, 'manual_labeled_data')
    original_data_path = os.path.join(integrate_root, 'original_labeled_data.json')
    with open(original_data_path, 'r') as f:
        original_data = json.load(f)
    manual_labeled_path = os.path.join(integrate_root, 'manual_labeled_data.json')
    with open(manual_labeled_path, 'r') as f:
        manual_data = json.load(f)

    integrate_data = {
        k: {
            'image_name': v['image_name'],
            'label': v['label'],
            'pseudo_label': {"缺陷": [], "管道附属设备及管配件": []},
            'prob': {"缺陷": [], "管道附属设备及管配件": []}
        }
        for k, v in original_data.items() if k != 'split'
    }
    last_id = len(integrate_data)
    for n, (k, v) in enumerate(manual_data.items()):
        value = {
            'image_name': v['image_name'],
            'label': v['label'],
            'pseudo_label': {"缺陷": [], "管道附属设备及管配件": []},
            'prob': {"缺陷": [], "管道附属设备及管配件": []}
        }
        if 'pseudo_label' in v.keys():
            value['candidate_label'] = v['pseudo_label']
        integrate_data[str(last_id + n)] = value

    integrate_path = os.path.join(integrate_root, 'integrate_data.json')
    with open(integrate_path, 'w') as f:
        f.write(json.dumps(integrate_data, indent=4, ensure_ascii=False))
    f.close()
    # distributor = Distributor(distribute_root, integrate_root)
    # distributor.clean_root()
    # distributor.max_prob = torch.rand(size=(1928,))
    # distributor.pre_label = torch.ones(size=(1928,))
    # distributor.distribute(n=3)
    # distributor.integrate_credible_label()
    # distributor.integrate_manual_label()


class WaterPiplinePredictor(object):
    def __init__(self, model_path=r'water_pipline_logs', integrate_root=r'D:\Public\water_pipeline'):
        self.model_path = model_path
        self.integrate_root = integrate_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.dic_dataset = {}
        self.data_loader = None

    def load_model(self, key):
        param_path = os.path.join(self.model_path, '{}_model_params.json'.format(key))
        with open(param_path, 'r') as data_file:
            param = json.load(data_file)
        self.model = ResNet(**param).to(self.device)
        print('load model from:{}'.format(param_path))

    def load_weights(self, key):
        path_mod = os.path.join(self.model_path, 'prune-*{}-*.pt')
        weights_path = glob.glob(path_mod.format(key))
        return weights_path

    def load_dataset(self, dataset_path):
        with open(dataset_path, 'r') as data_file:
            data_dic = json.load(data_file)
        self.dic_dataset = {k: {'image_name': v['image_name'],
                                'label': v['label'] if 'label' in v.keys() else {"缺陷": [], "管道附属设备及管配件": []},
                                'prob': {"缺陷": [], "管道附属设备及管配件": []}}
                            for k, v in data_dic.items()}
        dataset = {
            'image': np.array([v['image_name'] for v in self.dic_dataset.values()]),
            'id': np.array(list(self.dic_dataset.keys()))
        }
        dataset = SSLDataset(dataset, input_shape=(256, 512), random=['valid'])
        self.data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=64, shuffle=False, pin_memory=True,
            num_workers=4, persistent_workers=True, prefetch_factor=2
        )

    def assemble_predict(self, image, key='缺陷'):
        if isinstance(image, str) and image.split('.')[-1] in ('jpg', 'png', 'jpeg'):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        x = self.transform(image).to(torch.float32).to(self.device)
        probs = []
        param_path = self.load_weights(key=key)
        for path in param_path:
            weights = torch.load(path)
            self.model.load_state_dict(weights, strict=True)
            self.model.eval()
            with torch.no_grad():
                y = self.model(x)
                probs.append(torch.softmax(y, dim=1))
        return probs

    def predict_on_dataset(self, key='缺陷'):
        param_path = self.load_weights(key=key)
        dataset = self.data_loader.dataset
        for path in param_path:
            print("find weights: {}".format(path))
            self.model = torch.load(path).to(self.device)
            probs = []
            # self.model.load_state_dict(weights, strict=True)
            self.model.eval()
            with torch.no_grad():
                for x in self.data_loader:
                    x = x[0].to(self.device)
                    y_pre = self.model(x)
                    prob = torch.softmax(y_pre, dim=1)
                    probs.append(prob.cpu().numpy())
            probs = np.concatenate(probs, axis=0)
            probs = probs.tolist()
            for i in range(len(probs)):
                id = dataset.data['id'][i]
                self.dic_dataset[id]['prob'][key].append(probs[i])

    def save_prob(self, path=None):
        if path is None:
            path = os.path.join(self.integrate_root, 'integrate_data.json')
        with open(path, 'w') as file:
            file.write(json.dumps(self.dic_dataset, indent=4, ensure_ascii=False))
        file.close()

    def to_model_file(self, original_path=r'water_pipline_logs', index=None):
        file_path = self.model_path
        if os.path.exists(file_path) is False:
            os.mkdir(file_path)
        path_mod = 'ResNetAtt_0-3-4-6-4-all-*'
        model_path = glob.glob(os.path.join(original_path, path_mod))
        if index is not None:
            model_path = np.array(model_path)[index]
        for path in model_path:
            param_path = os.path.join(path, 'model_params.json')
            weights_path = glob.glob(os.path.join(path, '*.pt'))[-1]
            with open(param_path, 'r') as data_file:
                param = json.load(data_file)
            data_file.close()
            model = ResNetAtt(**param)
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            new_name = path.split('\\')[-1].replace('ResNetAtt_0-3-4-6-4-', '') + '.pt'
            print('{} to {}'.format(weights_path, new_name))
            torch.save(model, os.path.join(file_path, new_name))


if __name__ == '__main__':
    """
    dataset_path = r'datasets/cifar10'
    predictor = Predictor(None, dataset_path, None)
    image, label = predictor.dataset.__getitem__(10)
    image = torch.permute(image, (1, 2, 0)).cpu().detach().numpy()
    predictor.show_results(image, label=label)

    predictor.matrix = ((np.random.uniform(size=(5, 5)) + np.eye(5)) * 100).astype(np.uint8)
    predictor.show_matrix()
    """
    # meaningful_t()
    predictor = WaterPiplinePredictor(model_path=r'inference_param', integrate_root=r'D:\Public\water_pipeline')

    predictor.to_model_file(original_path=r'water_pipline_logs', index=[0])
    """
    predictor.load_dataset(dataset_path=r'D:\Public\water_pipeline\integrate_data.json')
    predictor.load_model(key='缺陷')
    predictor.predict_on_dataset(key='缺陷')
    predictor.load_model(key='管道附属设备及管配件')
    predictor.predict_on_dataset(key='管道附属设备及管配件')
    predictor.save_prob(path=r'D:\Public\water_pipeline\integrate_data_less_prob.json')
    """
