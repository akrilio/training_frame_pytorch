import torch
import numpy as np
import torch.utils.data as Data

from model.resnet import ResNet, ResNetAtt
from model.vits import UViT
from model.train import TrainerMixPrecise, TrainerFixMatch
from model.datasets import BasicDataset, SSLDataset
from model.loss_fn import evaluate, cross_entropy, SimLoss, ConsistencyLoss
from dataset_edit import DatasetEditor
from predict import Predictor


def supervise_learning():
    print('loading dataset ...')
    editor = DatasetEditor(dataset_path=r'datasets\cifar10')
    data = editor.read_cifar()

    # idx = np.concatenate([i[:400] for i in data['label_idx'].values()])
    # labeled_data = {key: np.array(value)[idx] for key, value in data['train'].items()}
    train_dataset = BasicDataset(data['train'], (32, 32), 'strong')
    valid_dataset = BasicDataset(data['valid'], (32, 32), 'valid')

    step_per_epoch = 500
    batch_size = 256
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_dataset, num_samples=batch_size * step_per_epoch, replacement=True)
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=256, shuffle=False, pin_memory=True,
        num_workers=2, persistent_workers=True, prefetch_factor=2,
    )
    print('creating model ...')
    """
    param = {'block_name': 'BottleNeck', 'num_block': [3, 4, 6, 3], 'num_classes': 10}
    model = ResNet(**param)
    
    params = {'image_size': (32, 32), 'channels': 3, 'classes': 10,
              'patch_size': (8, 8), 'depth': 12, 'dims': 384,
              'num_heads': 6, 'dim_heads': 64, 'mlp_ratio': 4,
              'dropout': 0.1, 'emb_dropout': 0.1}
    model = BasicViT(**params)
    
    params = {'image_size': (32, 32), 'patch_size': (2, 2), 'channels': 3, 'classes': 10,
              'depth': (2, 2, 3), 'dims': (96, 192, 384), 'scales': (1., 0.5, 0.25),
              'num_heads': (2, 4, 8), 'dim_heads': (48, 48, 48), 'mlp_ratio': 4,
              'dropout': 0., 'emb_dropout': 0., 'use_rel_pos': False}
    """
    param = {'block_name': 'BottleNeck', 'num_block': [3, 4, 6, 3], 'num_classes': 10}
    model = ResNet(**param)
    model.name += '_50'
    """
    path_mod = r'water_pipline_logs/ResNetAtt_0-3-4-6-4_self-supervise-*/checkpoint.pt'
    model_path = glob.glob(path_mod)[-1]
    weights = torch.load(model_path)
    weights.pop('output.weight')
    weights.pop('output.bias')
    model.load_state_dict(weights, strict=False)
    """
    torch.multiprocessing.set_start_method('spawn', force=True)
    print('setting trainer ...')
    train = TrainerMixPrecise(
        model=model, dataloader=(train_data_loader, None, valid_data_loader), loss=cross_entropy,
        lr=1e-4, weight_decay=1e-4, reduce_patience=8, stop_patience=200,
        verbose=True, callbacks={'func': evaluate, 'keys': ['precise', 'recall', 'mAP']},
        log_dir=r'water_pipline_logs', log_scalar=True, log_param=False
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
    train.run(num_epochs=300, step_per_epochs=step_per_epoch)


def semi_supervise_learning():
    print('loading dataset ...')
    editor = DatasetEditor(dataset_path=r'datasets\cifar10')
    data = editor.read_cifar()

    idx = np.concatenate([np.random.choice(i, size=400, replace=False) for i in data['label_idx'].values()])  # i[:400]
    labeled_data = {key: np.array(value)[idx] for key, value in data['train'].items()}
    train_l_dataset = BasicDataset(labeled_data, (32, 32), 'strong')
    train_ul_dataset = SSLDataset(data['train'], (32, 32), ['weak', 'strong'])
    valid_dataset = BasicDataset(data['valid'], (32, 32), 'valid')

    step_per_epoch = 500
    train_l_loader = Data.DataLoader(
        dataset=train_l_dataset, batch_size=32, shuffle=False, pin_memory=True,
        num_workers=2, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_l_dataset, num_samples=32 * step_per_epoch, replacement=True)
    )
    train_ul_loader = Data.DataLoader(
        dataset=train_ul_dataset, batch_size=128, shuffle=False, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_ul_dataset, num_samples=128 * step_per_epoch, replacement=True)
    )
    valid_loader = Data.DataLoader(
        dataset=valid_dataset, batch_size=512, shuffle=False, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
    )
    print('creating model ...')
    param = {'block_name': 'BottleNeck', 'num_block': [3, 4, 6, 3], 'num_classes': 10}
    model = ResNet(**param)
    model.name += '_50_refix-match'
    print('setting trainer ...')
    fix_match_loss = {'sup_loss': cross_entropy, 'unsup_loss': ConsistencyLoss(b=32, T=0.5, p_cutoff=0.95)}
    torch.multiprocessing.set_start_method('spawn', force=True)
    train = TrainerFixMatch(
        model=model, dataloader=(train_l_loader, train_ul_loader, valid_loader),
        loss=fix_match_loss, lr=1e-3, weight_decay=1e-4, reduce_patience=8, stop_patience=500,
        verbose=True, callbacks={'func': evaluate, 'keys': ['precise', 'recall', 'mAP']},
        log_dir=r'logs', log_scalar=True, log_param=False
    )
    print('running ...')
    train.run(num_epochs=150, step_per_epochs=step_per_epoch)


def self_supervise_learning():
    print('loading dataset ...')
    editor = DatasetEditor(dataset_path=r'datasets\cifar10')
    data = editor.read_cifar()
    train_ul_dataset = SSLDataset(data['train'], (32, 32), ['strong', 'strong'])

    step_per_epoch = 500
    batch_size = 64
    train_ul_loader = Data.DataLoader(
        dataset=train_ul_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=12, persistent_workers=True, prefetch_factor=2,
        sampler=Data.RandomSampler(train_ul_dataset, num_samples=batch_size * step_per_epoch, replacement=True)
    )
    print('creating model ...')

    param = {'input_shape': (256, 512), 'num_classes': 512,
             'encoder_param': {
                 'block_name': 'BasicBlock', 'num_block': (3, 4, 6),
                 'channels': (32, 64, 128), 'first_stride': 2,
                 'expansion': 1, 'dropout': 0.
             },
             'decoder_param': {
                 'block_name': 'LinearAttV2', 'inner_shape': (2048, 2048),
                 'num_block': 4, 'channels': 512, 'convert_stride': 2,
                 'dropout': 0.1
             }
             }
    model = ResNetAtt(**param)
    model.name += '_0-3-4-6-4_self-supervise'
    torch.multiprocessing.set_start_method('spawn', force=True)
    # weights = torch.load(r'water_pipline_logs/UViT2-4-8_2-2-3_self-supervise-20240915-094427/checkpoint.pt')
    # model.load_state_dict(weights, strict=True)
    print('setting trainer ...')
    sim_loss = SimLoss(batch_size=batch_size)
    train = TrainerMixPrecise(
        model=model, dataloader=(train_ul_loader, None, None), loss=sim_loss,
        lr=1e-4, weight_decay=1e-4, reduce_patience=8, stop_patience=200,
        verbose=True, callbacks=None, log_dir=r'water_pipline_logs', log_scalar=True, log_param=False
    )
    print('running ...')
    train.run(num_epochs=300)


def eval_model():
    model_path = r'logs/20240420-170918'
    dataset_path = r'datasets/dataset_6_2_2'
    predictor = Predictor(ResNet, dataset_path, model_path)

    predictor.test_on_dataset()
    predictor.show_matrix(save=False)
    """
    image_data, label_data = predictor.dataset.__getitem__(10)
    image = torch.permute(image_data, (1, 2, 0)).cpu().detach().numpy()
    label = torch.permute(label_data, (1, 2, 0)).cpu().detach().numpy()
    pre = predictor(image, label, show=True, use_crf=False, use_region=False)
    """


def clean():
    import gc
    all_objects = gc.get_objects()
    for obj in all_objects:
        del obj


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    supervise_learning()
    # semi_supervise_learning()
    clean()
