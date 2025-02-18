import os

import matplotlib.pyplot as plt
import numpy as np
import glob
import json
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import cv2


class Searcher(object):
    def __init__(self, root_path=None, dataset_path=r'/home/a430/LXQ/water_pipeline'):
        self.root_path = root_path
        if os.path.exists(dataset_path) is False and root_path is not None:
            os.mkdir(dataset_path)
        self.dataset_path = dataset_path
        self.labels = {
            '缺陷': {
                0: {'formal_name': '正常ZC', 'full_names': ['正常']},
                1: {'formal_name': '泄漏XL', 'up_level': '结构性缺陷SD',
                    'full_names': ['泄漏', '漏点']},
                2: {'formal_name': '腐蚀FS', 'up_level': '结构性缺陷SD',
                    'full_names': ['腐蚀', '脱落', '脱离']},
                3: {'formal_name': '管瘤GL', 'up_level': '结构性缺陷SD',
                    'full_names': ['管瘤', '结垢']},
                4: {'formal_name': '气囊QN', 'up_level': '功能性缺陷FD',
                    'full_names': ['气囊']},
                5: {'formal_name': '异物YW_or_杂质ZZ', 'up_level': '功能性缺陷FD',
                    'full_names': ['杂质', '悬浮', '沉积', '异物', '障碍', '淤塞', '於塞', '淤积', '残留']},
                # 7: {'formal_name': '不明连接BM', 'up_level': '功能性缺陷FD',
                #     'full_names': ['不明连接']},
            },
            '管道附属设备及管配件': {
                0: {'formal_name': '无None', 'up_level': '管道附属设备及管配件',
                    'full_names': ['正常', '变径管BJG', '变径', '流量仪LLY', '流量仪', '流量计']},
                1: {'formal_name': '支管ZG', 'up_level': '管道附属设备及管配件',
                    'full_names': ['支管', '分支', '人孔RK', '人孔']},
                2: {'formal_name': '弯头WT', 'up_level': '管道附属设备及管配件',
                    'full_names': ['弯头', '弯管', '弯道']},
                3: {'formal_name': '阀门FM', 'up_level': '管道附属设备及管配件',
                    'full_names': ['阀门', '蝶阀', '腰阀', '闷板', '尽头']},
            }
        }

    def search_label(self, name):
        label = {'缺陷': 0, '管道附属设备及管配件': 0}
        for key, sub_label in self.labels.items():
            for k, v in sub_label.items():
                for n in v['full_names']:
                    if name.find(n) >= 0:
                        label[key] = k
        if label['缺陷'] == 0 and label['管道附属设备及管配件'] == 0:
            print(name)
        return label

    def search_images(self, search):
        search_path = glob.glob(search)
        name_1 = [p.split("\\")[4] for p in search_path]
        name_2 = [p.split("\\")[-1].replace(" ", '') for p in search_path]
        labels = [self.search_label(n) for n in name_2]
        image_name = [n1 + n2 for n1, n2 in zip(name_1, name_2)]
        return search_path, labels, image_name

    def create_original_labeled_dataset(self):
        # F:\供水管道\供水管网检测\4.北苏州路\1北苏州路（浙江北路-西藏北路）A\图片\连接及附件
        search1 = os.path.join(self.root_path, "供水管网检测", '*', '*', '*', '*.jpg')
        search_path1, labels1, image_name1 = self.search_images(search1)
        search2 = os.path.join(self.root_path, "供水管网检测", '*', '*', '*', '*', '*.jpg')
        search_path2, labels2, image_name2 = self.search_images(search2)
        search_path = search_path1 + search_path2
        labels = labels1 + labels2
        image_name = image_name1 + image_name2
        image_information = {
            str(i): {'image_name': os.path.join(self.dataset_path, 'original_labeled_data', "ID{}_{}".format(i, ph)),
                     'label': lb}
            for i, (ph, lb) in enumerate(zip(image_name, labels))
        }
        print(len(list(image_information.keys())))
        """"""
        if os.path.exists(os.path.join(self.dataset_path, 'original_labeled_data')) is False:
            os.mkdir(os.path.join(self.dataset_path, 'original_labeled_data'))
        for s_p, v in zip(search_path, image_information.values()):
            image = Image.open(s_p)
            image.save(v['image_name'])
        with open(os.path.join(self.dataset_path, "original_labeled_data.json"), 'w') as write_f:
            write_f.write(json.dumps(image_information, indent=4, ensure_ascii=False))
        write_f.close()

    def create_split_index(self, ratio=0.2):
        filename = os.path.join(self.dataset_path, 'original_labeled_data.json')
        with open(filename, 'r') as data_file:
            labeled_data = json.load(data_file)
        data_file.close()

        labeled_data.pop('split')
        labels = np.array([list(i['label'].values()) for i in labeled_data.values()])
        labels = 10 * (labels[:, 0] + 1) + (labels[:, 1] + 1)
        keys = np.array(list(labeled_data.keys()))

        labeled_data['split'] = {'train': [], 'valid': []}
        unique_label, counts = np.unique(labels, return_counts=True)
        for l, n in zip(unique_label, counts):
            idx = np.where(labels == l)[0]
            np.random.shuffle(idx)
            split = int(np.clip(n * ratio, a_min=1, a_max=np.inf))
            labeled_data['split']['train'] += keys[idx[:-split]].tolist()
            labeled_data['split']['valid'] += keys[idx[-split:]].tolist()
            if n == 1:
                labeled_data['split']['train'] += keys[idx].tolist()
        """"""
        with open(filename, 'w') as data_file:
            data_file.write(json.dumps(labeled_data, indent=4, ensure_ascii=False))
        data_file.close()

    def change_labels(self):
        label_path = os.path.join(self.dataset_path, "original_labeled_data.json")
        with open(label_path, 'r') as f:
            label_dic = json.load(f)
        f.close()
        """
        image_names = np.array([value['image_name'] for value in label_dic.values()])
        deform = np.array([value['label']['缺陷'] for value in label_dic.values()])
        struc = np.array([value['label']['管道附属设备及管配件'] for value in label_dic.values()])
        idx = np.where(struc == 4)[0]
        image_name = image_names[idx]
        fig, axes = plt.subplots(1, min(len(image_name), 4))
        if len(image_name) == 1:
            axes.imshow(Image.open(image_name[0]))
            axes.set_title(os.path.basename(image_name[0]))
        else:
            for ax, name in zip(axes.flatten(), image_name[4:8]):
                ax.imshow(Image.open(name))
                ax.set_title(os.path.basename(name))
        plt.show()
        """
        for value in label_dic.values():
            name = value['image_name'].split('\\')[-1]
            value['label'] = self.search_label(name)
        with open(label_path, 'w') as write_f:
            write_f.write(json.dumps(label_dic, indent=4, ensure_ascii=False))
        write_f.close()

    def save_label_name(self):
        label_name_path = os.path.join(self.dataset_path, "label_information.json")
        with open(label_name_path, 'w') as write_f:
            write_f.write(json.dumps(self.labels, indent=4, ensure_ascii=False))
        write_f.close()

    def video_to_image(self, name, frame_interval=200):
        print(f"Processing {name} ...")
        save_path = os.path.join(self.dataset_path, 'unlabeled_data', name.split('\\')[-1])
        image_names = []
        try:
            video = cv2.VideoCapture(name)
        except Exception as e:
            print(f"Error opening video file {name}: {str(e)}")
            return image_names
        count = 0
        id = 0
        while True:
            try:
                success, frame = video.read()
            except Exception as e:
                print(f"Error processing frame {count} in {name}: {str(e)}")
                continue
            if not success:
                break
            if count % frame_interval == 0:
                image_name = save_path.replace('.mp4', f"_{id}.jpg")
                cv2.imencode('.jpg', frame)[1].tofile(image_name)
                image_names.append(image_name)
                id += 1
            count += 1
        video.release()
        print(f"Processed {name}: extracted {len(image_names)} frames")
        return image_names

    def create_unlabeled_dataset(self):
        # F:\供水管道\供水视频数据\1.爱辉路\A爱辉路（呼兰路-蕰藻浜）\管内作业视频
        search = os.path.join(self.root_path, "供水视频数据", '*', '*', '*', '*.mp4')
        search_path = glob.glob(search)
        search = os.path.join(self.root_path, "供水视频数据", '*', '*', '*', '*', '*.mp4')
        search_path += glob.glob(search)

        names = np.array([p.split('\\')[-1].replace('.mp4', '') for p in search_path])

        unlabeled_dataset_path = os.path.join(self.dataset_path, 'unlabeled_data.json')
        with open(unlabeled_dataset_path, 'r') as file:
            unlabeled_dic = json.load(file)
        file.close()
        loaded_names = set(os.path.splitext(os.path.basename(value['image_name']))[0]
                           for value in unlabeled_dic.values())
        new_path = [path for path, name in zip(search_path, names) if name not in loaded_names]
        with ProcessPoolExecutor(max_workers=32) as executor:
            unlabeled_images = list(executor.map(self.video_to_image, new_path))
        unlabeled_images = [item for sublist in unlabeled_images for item in sublist]

        n = len(list(unlabeled_dic.keys()))
        print('new_number: {}, original_number: {}, new_path: {}'
              .format(len(unlabeled_images), len(list(unlabeled_dic.keys())), len(new_path)))
        print(np.array(new_path))
        unlabeled_data = {
            i: {'image_name': ui,
                'pseudo_label': {'缺陷': [], '管道附属设备及管配件': []},
                'prob': {'缺陷': [], '管道附属设备及管配件': []}}
            for i, ui in enumerate(unlabeled_images, n)
        }
        unlabeled_dic.update(unlabeled_data)
        print('total_number: {}'.format(len(list(unlabeled_dic.keys()))))
        
        unlabeled_name_path = os.path.join(self.dataset_path, 'unlabeled_data.json')
        with open(unlabeled_name_path, 'w') as write_f:
            write_f.write(json.dumps(unlabeled_dic, indent=4, ensure_ascii=False))
        write_f.close()
        """"""

    def clean_unused_unlabeled_data(self):
        unlabeled_dataset_path = os.path.join(self.dataset_path, 'unlabeled_data.json')

        with open(unlabeled_dataset_path, 'r') as file:
            unlabeled_dic = json.load(file)
        file.close()
        used_image_names = set(value['image_name'] for value in unlabeled_dic.values())
        print(len(used_image_names))
        search = os.path.join(self.dataset_path, 'unlabeled_data', '*')
        image_names = glob.glob(search)
        unused_image_names = [image_name for image_name in image_names if image_name not in used_image_names]
        """"""
        for names in unused_image_names:
            os.remove(names)

    def resize_images(self, size, image_dirs=('original_labeled_data', 'unlabeled_data')):
        for image_dir in image_dirs:
            ori_dir = os.path.join(self.dataset_path, image_dir)
            new_dir = ori_dir + '_resized'
            if os.path.exists(new_dir) is False:
                os.mkdir(new_dir)
            image_names = glob.glob(os.path.join(ori_dir, '*.jpg'))
            for image_name in image_names:
                image = Image.open(image_name).resize(size)
                image.save(image_name.replace(ori_dir, new_dir))

    def equal_distribute(self, dataset, T=100):
        image_name = dataset['image']
        label = dataset['label']
        unique_label, num = np.unique(label, return_counts=True)
        if isinstance(T, int):
            ratio = np.log(num / T + 1)
            ratio = ratio / sum(ratio)
        elif isinstance(T, (list, tuple)):
            ratio = T
        else:
            ratio = [1 / len(unique_label)] * len(unique_label)
        equaled_num = (max(num / ratio) * ratio).astype(np.uint)
        print('equalized label ratio: {}'.format(equaled_num))
        equaled_image_name = np.concatenate([
            image_name[np.random.choice(np.where(label == l)[0], size=n)]
            for l, n in zip(unique_label, equaled_num)
        ], axis=0)
        equaled_label = np.concatenate([
            np.repeat(l, (n,))
            for l, n in zip(unique_label, equaled_num)
        ], axis=0)
        return {'image': equaled_image_name, 'label': equaled_label}

    def to_one_hot(self, label):
        label = label.astype(np.int64)
        one_hot_label = []
        for l in label.T:
            mod = np.eye(np.max(l))
            one_hot_label.append(mod[l])
        return np.concatenate(one_hot_label, axis=-1)

    def read_dataset(self, label=None, resized=False, equal=True, T=100, one_hot=False):
        labeled_data = {}

        labeled_data_path = os.path.join(self.dataset_path, 'original_labeled_data.json')
        with open(labeled_data_path, 'r') as data_file:
            original_labeled_data = json.load(data_file)
        data_file.close()

        if label == 'all':
            items = np.array([
                [os.path.join(self.dataset_path, v['image_name']),]
                + list(v['label'].values())
                for k, v in original_labeled_data.items() if k != 'split'
            ])
            image_names, labels = items[:, 0], items[:, 1:]
        else:
            items = np.array([
                [v['image_name'], v['label'][label]]
                for k, v in original_labeled_data.items() if k != 'split'
            ])
            image_names, labels = items.T
        if resized:
            for image_name in image_names:
                image_name.replace('original_labeled_data', 'original_labeled_data_resized')
        labels = labels.astype(np.uint)
        train_idx = np.array(original_labeled_data['split']['train'], dtype=np.uint)
        labeled_data['train'] = {'image': image_names[train_idx], 'label': labels[train_idx]}
        valid_idx = np.array(original_labeled_data['split']['valid'], dtype=np.uint)
        labeled_data['valid'] = {'image': image_names[valid_idx], 'label': labels[valid_idx]}
        """"""
        labeled_data_path = os.path.join(self.dataset_path, 'manual_labeled_data.json')
        if os.path.exists(labeled_data_path):
            with open(labeled_data_path, 'r') as data_file:
                manual_labeled_data = json.load(data_file)
            data_file.close()
            if label == 'all':
                items = np.array([
                    [os.path.join(self.dataset_path, v['image_name']),]
                    + list(v['label'].values())
                    for k, v in manual_labeled_data.items() if k != 'split'
                ])
                image_names, labels = items[:, 0], items[:, 1:]
            else:
                items = np.array([
                    [os.path.join(self.dataset_path, v['image_name']),
                     v['label'][label]]
                    for k, v in manual_labeled_data.items() if k != 'split'
                ])
                image_names, labels = items.T
            if resized:
                for image_name in image_names:
                    image_name.replace('unlabeled_data', 'unlabeled_data_resized')
            labels = labels.astype(np.uint)
            labeled_data['train']['image'] = np.concatenate(
                [labeled_data['train']['image'], image_names], axis=0
            )
            labeled_data['train']['label'] = np.concatenate(
                [labeled_data['train']['label'], labels], axis=0
            )
        label_name_path = os.path.join(self.dataset_path, 'label_information.json')
        with open(label_name_path, 'r', encoding='GBK') as data_file:
            if label == 'all':
                labeled_data['label_names'] = json.load(data_file)
            else:
                labeled_data['label_names'] = json.load(data_file)[label]
        data_file.close()

        unlabeled_data_path = os.path.join(self.dataset_path, 'unlabeled_data.json')
        with open(unlabeled_data_path, 'r') as data_file:
            unlabeled_data_dic = json.load(data_file)
        data_file.close()
        unlabeled_data = {
            'image': np.array([os.path.join(self.dataset_path, i['image_name'])
                               for i in unlabeled_data_dic.values()]),
            'id': np.array(list(unlabeled_data_dic.keys()))
        }
        if resized:
            for image_name in unlabeled_data['image']:
                image_name.replace('unlabeled_data', 'unlabeled_data_resized')
        if equal:
            labeled_data['train'] = self.equal_distribute(labeled_data['train'], T)
            labeled_data['valid'] = self.equal_distribute(labeled_data['valid'], T)
        if one_hot:
            for value in labeled_data.values():
                value['label'] = self.to_one_hot(value['label'])
        return {'train': labeled_data['train'], 'valid': labeled_data['valid'], 'unlabeled': unlabeled_data}

    def change_image_path(self):
        transform_paths = [
            'original_labeled_data.json',
            'manual_labeled_data.json',
            'unlabeled_data.json',
        ]
        for path in transform_paths:
            abs_path = os.path.join(self.dataset_path, path)
            print(abs_path)
            with open(abs_path, 'r') as data_file:
                datas = json.load(data_file)
            data_file.close()
            for k, v in datas.items():
                if k != 'split':
                    image_path = v['image_name']
                    image_path = image_path.replace('D:\\Public\\water_pipeline\\', '')
                    image_path = image_path.replace('\\', '/')
                    datas[k]['image_name'] = image_path
            with open(abs_path, 'w') as f:
                f.write(json.dumps(datas, indent=4, ensure_ascii=False))
            f.close()
            print("{}, transfer complete".format(abs_path))


if __name__ == '__main__':
    searcher = Searcher()
    # searcher.change_image_path()
    # searcher.resize_images(size=(512, 256))
    # searcher.create_original_labeled_dataset()
    # searcher.change_labels()
    # searcher.save_label_name()
    # searcher.create_split_index(ratio=0.2)
    # searcher.create_unlabeled_dataset()
    # searcher.clean_unused_unlabeled_data()

    dataset = searcher.read_dataset(label='all', resized=False, equal=False, T=100)
    for k, v in dataset['train'].items():
        print(k, v.shape)
        if k == 'label':
            print(np.unique(v[:, 0], return_counts=True))
            print(np.unique(v[:, 1], return_counts=True))
            print(np.unique(10 * v[:, 0] + v[:, 1], return_counts=True))
        else:
            print(v[-5:])
    for key, value in dataset['unlabeled'].items():
        print(key, value.shape)
    """
    from model.datasets import BasicDataset
    dataset = BasicDataset(dataset['train'], (256, 512), 'strong')
    idx = (np.random.random(size=(4,)) * len(dataset)).astype(np.uint)
    fig, axes = plt.subplots(2, 2)
    for i, ax in zip(idx, axes.flatten()):
        image, label = dataset[i]
        ax.imshow(image.permute((1, 2, 0)).numpy())
        ax.set_title("{}".format(dataset.data['image'][i]))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()
    
    image_path = glob.glob(r'D:\Public\water_pipelineunlabeled_data\*.jpg')
    name = [os.path.basename(path).split('_') for path in image_path]
    dn = []
    for n in name:
        for d in n:
            if "DN" in d:
                dn.append(d)
    print(np.unique(np.array(dn)))
    # 'DN1000' 'DN1200' 'DN1500' 'DN1800' 'DN300' 'DN500' 'DN600' 'DN700' 'DN800' 'DN900'
    """