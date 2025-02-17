import glob

import torch
import os
import torch.utils.data as Data
# from torchvision.transforms import v2

from model.resnet_light import ResNetAtt
from model.train import TrainerFixMatch, TrainerSimCLR, TrainerMixPrecise
from model.datasets import BasicDataset, SSLDataset
from model.loss_fn import evaluate, cross_entropy, ConsistencyLoss, SimLoss
from predict import Distributor
from image_searcher import Searcher


def supervise_learning(label=None):
    print('loading dataset ...')
    searcher = Searcher(root_path=r'F:\供水管道', dataset_path=r'D:\Public\water_pipeline')
    lb_dataset, _ = searcher.read_dataset(label=label, equal=False, T=10)
    train_l_dataset = BasicDataset(lb_dataset['train'], (256, 512), 'strong')
    valid_dataset = BasicDataset(lb_dataset['valid'], (256, 512), 'valid')

    step_per_epoch = 250
    batch_size = 128

    train_loader = Data.DataLoader(
        dataset=train_l_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=16, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_l_dataset, num_samples=batch_size * step_per_epoch, replacement=True)
    )
    valid_loader = Data.DataLoader(
        dataset=valid_dataset, batch_size=256, shuffle=False, pin_memory=True,
        num_workers=2, persistent_workers=True, prefetch_factor=2,
    )
    print('creating model ...')
    """
    param = {'input_shape': (256, 512), 'num_classes': 10,
             'encoder_param': {
                 'block_name': 'BasicBlockPreNorm', 'num_block': (3, 4, 6),
                 'channels': (32, 64, 128), 'first_stride': 2,
                 'expansion': 2, 'dropout': 0.
             },
             'decoder_param': {
                 'block_name': 'LinearAttSa', 'inner_shape': (2048, 2048),
                 'num_block': 4, 'channels': 512, 'convert_stride': 2,
                 'dropout': 0.
             }
            }
    """
    param = {'input_shape': (256, 512), 'num_classes': 10,
             'encoder_param': {
                 'block_name': ['ResPreNorm', 'ConvSeqLight'],
                 'num_block': (2, 2, 3), 'channels': (32, 64, 128),
                 'first_stride': 2, 'expansion': 2, 'dropout': 0.
             },
             'decoder_param': {
                 'block_name': ['SpaceGateRes', 'LinearResPreNorm', 'CTLinearSeq'],
                 'num_block': 7, 'convert_stride': 2, 'channel': 512,
                 'hidden_channel': 2048, 'num_head': 4, 'dropout': 0.,
             },
             'head': 'AttnHead',
             }
    model = ResNetAtt(**param)
    model.name += '_0-2-2-3-7-' + label + '-no-pretrain'
    """
    # path_mod = r'water_pipline_logs/ResNetAtt_0-3-4-6-4-{}-*/last_epoch_*.pt'.format(label)
    path_mod = r'water_pipline_logs/ResNetAtt_0-2-2-3-7_self-supervise-*/checkpoint.pt'
    # path_mod = r'water_pipline_logs/LinearTf_7_self-supervise-*/checkpoint.pt'
    model_path = glob.glob(path_mod)[-1]
    weights = torch.load(model_path)
    weights.pop('head.attn.0.weight')
    weights.pop('head.attn.0.bias')
    model.load_state_dict(weights, strict=False)
    """
    torch.multiprocessing.set_start_method('spawn', force=True)
    print('setting trainer ...')
    weight = 1 / torch.tensor([1., 1., 0.3, 0.3, 0.1, 0.2], device='cuda')  # [0.72, 1, 0.46, 0.60, 0.1, 0.32]
    weight_0 = 6 * weight / torch.sum(weight)
    weight = 1 / torch.tensor([1., 0.1, 0.1, 0.1], device='cuda')  # [0.92, 0.20, 0.30, 0.46]
    weight_1 = 4 * weight / torch.sum(weight)

    def cross_entropy_wrap(y_true, y_pred):
        loss_0 = cross_entropy(y_true[:, 0], y_pred[:, [0, 1, 2, 3, 4, 5]], weight=weight_0)
        loss_1 = cross_entropy(y_true[:, 1], y_pred[:, [6, 7, 8, 9]], weight=weight_1)
        return loss_0 + loss_1

    def evaluate_warp(y_true, y_pred):
        loss_0 = evaluate(y_true[:, 0], y_pred[:, [0, 1, 2, 3, 4, 5]])
        loss_1 = evaluate(y_true[:, 1], y_pred[:, [6, 7, 8, 9]])
        return loss_0 + loss_1

    callbacks = {
        'func': evaluate_warp,
        'keys': ['precise_0', 'recall_0', 'mAP', 'precise_1', 'recall_1', 'mAP']  # ['precise', 'recall', 'mAP']
    }
    train = TrainerMixPrecise(
        model=model, dataloader=(train_loader, None, valid_loader), loss=cross_entropy_wrap,
        lr=1e-4, weight_decay=1e-2, reduce_patience=8, stop_patience=200,
        verbose=True, callbacks=callbacks, log_dir=r'water_pipline_logs', log_scalar=True, log_param=False
    )
    """
    cutmix = v2.CutMix(num_classes=len(lb_dataset['label_names']))
    mixup = v2.MixUp(num_classes=len(lb_dataset['label_names']))
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    def unpack(data):
        x, y = data
        x, y = cutmix_or_mixup(x, y)
        return x.to(device), y.to(device)
    train.unpack = unpack
    """
    print('running ...')
    train.run(num_epochs=200, step_per_epochs=step_per_epoch)
    """
    for param in train.model.body.parameters():
        param.requires_grad = False
    train.run(num_epochs=2, step_per_epochs=step_per_epoch)
    for param in train.model.body.parameters():
        param.requires_grad = True
    train.run(num_epochs=200, step_per_epochs=step_per_epoch)
    """


def semi_supervise_learning(label=None):
    print('loading dataset ...')
    searcher = Searcher(root_path=r'F:\供水管道', dataset_path=r'D:\Public\water_pipeline')
    lb_dataset, ulb_dataset = searcher.read_dataset(label=label, equal=True, T=100)
    train_l_dataset = BasicDataset(lb_dataset['train'], (256, 512), 'strong')
    train_ul_dataset = SSLDataset(ulb_dataset, (256, 512), ['weak', 'strong'])
    valid_dataset = BasicDataset(lb_dataset['valid'], (256, 512), 'valid')

    step_per_epoch = 250
    batch_size = 8
    train_l_loader = Data.DataLoader(
        dataset=train_l_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_l_dataset, num_samples=batch_size * step_per_epoch, replacement=True)
    )
    train_ul_loader = Data.DataLoader(
        dataset=train_ul_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_ul_dataset, num_samples=batch_size * step_per_epoch, replacement=True)
    )
    valid_loader = Data.DataLoader(
        dataset=valid_dataset, batch_size=32, shuffle=False, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
    )
    print('creating model ...')
    param = {'input_shape': (256, 512), 'num_classes': 10,
             'encoder_param': {
                 'block_name': 'BasicBlockPreNorm', 'num_block': (3, 4, 6),
                 'channels': (32, 64, 128), 'first_stride': 2,
                 'expansion': 2, 'dropout': 0.
             },
             'decoder_param': {
                 'block_name': 'LinearAtt', 'inner_shape': (2048, 2048),
                 'num_block': 7, 'channels': 512, 'convert_stride': 2,
                 'dropout': 0.
             }
             }
    model = ResNetAtt(**param)
    model.name += '_0-3-4-6-4-semi-supervise' + label
    # path_mod = r'water_pipline_logs/ResNet_48_refix-match-{}*/last_epoch_*.pt'.format(label)
    path_mod = r'water_pipline_logs/ResNet_48_self-supervise-*/checkpoint.pt'
    model_path = glob.glob(path_mod)[1]
    print("find latest path: {}".format(model_path))
    weights = torch.load(model_path)
    weights.pop('fc.weight')
    weights.pop('fc.bias')
    model.load_state_dict(weights, strict=False)

    print('setting trainer ...')
    fix_match_loss = {'sup_loss': cross_entropy, 'unsup_loss': ConsistencyLoss(b=32, T=0.5, p_cutoff=0.95)}
    torch.multiprocessing.set_start_method('spawn', force=True)
    train = TrainerFixMatch(model=model, dataloader=(train_l_loader, train_ul_loader, valid_loader),
                            loss=fix_match_loss, lr=1e-3, weight_decay=1e-4, stop_patience=150,
                            verbose=True, callbacks={'func': evaluate, 'keys': ['precise', 'recall', 'mAP']},
                            log_dir=r'water_pipline_logs', log_scalar=True, log_param=False)
    print('running ...')
    train.run(num_epochs=200, step_per_epochs=step_per_epoch)
    """
    for param in train.model.body.parameters():
        param.requires_grad = False
    train.run(num_epochs=5, step_per_epochs=step_per_epoch)
    for param in train.model.body.parameters():
        param.requires_grad = True
    train.run(num_epochs=195, step_per_epochs=step_per_epoch)
    """


def self_supervise_learning():
    import numpy as np
    torch.multiprocessing.set_start_method('spawn', force=True)
    print('loading dataset ...')
    searcher = Searcher(root_path=r'F:\供水管道', dataset_path=r'D:\Public\water_pipeline')
    lb_dataset, ulb_dataset = searcher.read_dataset(label='缺陷', resized=False, equal=False)
    idx = np.arange(len(ulb_dataset['image']))
    split = int(idx[-1] * 0.1 // 64) * 64
    print('total sample:{}, {} for train, {} for valid'.format(idx[-1], idx[-1] - split, split))
    np.random.shuffle(idx)
    train_data = {key: value[idx[:-split]] for key, value in ulb_dataset.items()}
    valid_data = {key: value[idx[-split:]] for key, value in ulb_dataset.items()}
    train_ul_dataset = SSLDataset(train_data, (256, 512), ['strong', 'strong'])
    valid_ul_dataset = SSLDataset(valid_data, (256, 512), ['strong', 'strong'])
    step_per_epoch = 250
    batch_size = 64
    train_ul_loader = Data.DataLoader(
        dataset=train_ul_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=12, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_ul_dataset, num_samples=batch_size * step_per_epoch, replacement=True)
    )
    valid_ul_loader = Data.DataLoader(
        dataset=valid_ul_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=16, persistent_workers=True, prefetch_factor=2,
    )
    print('creating model ...')
    """
    param = {'input_shape': (256, 512), 'num_classes': 512,
             'encoder_param': {
                 'block_name': 'BasicBlockPreNorm', 'num_block': (3, 4, 6),
                 'channels': (32, 64, 128), 'first_stride': 2,
                 'expansion': 2, 'dropout': 0.
             },
             'decoder_param': {
                 'block_name': 'LinearAttSa', 'inner_shape': (2048, 2048),
                 'num_block': 4, 'channels': 512, 'convert_stride': 2,
                 'dropout': 0.
             }
             }
    """
    param = {'input_shape': (256, 512), 'num_classes': 512,
             'encoder_param': {
                 'block_name': ['ResPreNorm', 'ConvSeqLight'],
                 'num_block': (2, 2, 3), 'channels': (32, 64, 128),
                 'first_stride': 2, 'expansion': 2, 'dropout': 0.
             },
             'decoder_param': {
                 'block_name': ['LinearResPreNorm', 'SpaceGate', 'CTLinearSeq'],
                 'num_block': 7, 'convert_stride': 2, 'channel': 512,
                 'hidden_channel': 2048, 'num_head': 4, 'dropout': 0.,
             },
             'head': 'AttnHead',
             }
    model = ResNetAtt(**param)
    model.name += '_0-2-2-3-7_self-supervise'

    # weights = torch.load(r'water_pipline_logs/UViT2-4-8_2-2-3_self-supervise-20240915-094427/checkpoint.pt')
    # model.load_state_dict(weights, strict=True)
    print('setting trainer ...')
    sim_loss = SimLoss(batch_size=2 * batch_size)
    train = TrainerSimCLR(model=model, dataloader=(train_ul_loader, None, valid_ul_loader), loss=sim_loss,
                          lr=1e-4, weight_decay=1e-2, reduce_patience=8, stop_patience=150,
                          verbose=True, callbacks=None,
                          log_dir=r'water_pipline_logs', log_scalar=True, log_param=False)
    print('running ...')
    train.run(num_epochs=200)


def eval_model():
    # path_mod = r'water_pipline_logs/ResNet_50_{}*'
    path_mod = r'water_pipline_logs/ResNetAtt_2-4-6-3_4-{}-*'
    integrate_root = r'D:\Public\water_pipeline'
    """"""
    key = 'all'  # '缺陷'
    model_path = glob.glob(path_mod.format(key))[-2]
    print("find latest path: {}".format(model_path))
    distribute_root = os.path.join(integrate_root, 'manual_labeled_data')

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
                label_data[:, 1] += 6
                pre = torch.stack([torch.argmax(pre[:, :6], dim=-1),
                                   torch.argmax(pre[:, 6:], dim=-1) + 6], dim=-1)
                item = torch.stack([label_data, pre], dim=-1).reshape(-1, 2)
                for i in item:
                    self.matrix[i[0], i[1]] += 1

    Distributor.test_on_dataset = test_on_dataset
    distributor = Distributor(distribute_root=distribute_root, integrate_root=integrate_root,
                              model=ResNetAtt, model_path=model_path, image_shape=(256, 512), key=key)
    distributor.test_on_dataset()
    distributor.show_matrix()
    """
    distributor.clean_root()
    distributor.prob_predict()
    distributor.distribute(n=25)
    distributor.integrate_credible_label()
    
    key = '管道附属设备及管配件'
    model_path = glob.glob(path_mod.format(key))[-2]
    print("find latest path: {}".format(model_path))
    distribute_root = os.path.join(integrate_root, 'manual_labeled_data')
    distributor = Distributor(distribute_root=distribute_root, integrate_root=integrate_root,
                              model=ResNet, model_path=model_path, image_shape=(256, 512), key=key)
    distributor.prob_predict()
    distributor.distribute(n=25)
    distributor.integrate_credible_label()
    """


def collection():
    integrate_root = r'D:\Public\water_pipeline'
    distribute_root = os.path.join(integrate_root, 'manual_labeled_data')
    distributor = Distributor(distribute_root=distribute_root, integrate_root=integrate_root)
    distributor.integrate_manual_label()


def clean():
    import gc
    all_objects = gc.get_objects()
    for obj in all_objects:
        del obj


if __name__ == '__main__':
    # collection()
    # clean()
    # self_supervise_learning()
    supervise_learning(label='all')
    # supervise_learning(label='缺陷')
    # supervise_learning(label='管道附属设备及管配件')
    # semi_supervise_learning(label='缺陷')
    # semi_supervise_learning(label='管道附属设备及管配件')
    # clean()
    # eval_model()
