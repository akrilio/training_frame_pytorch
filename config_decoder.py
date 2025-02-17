import os
from ruamel.yaml import YAML
import types
import torch

from dataset_edit import DatasetEditor
from image_searcher import Searcher
from model.datasets import *
import torch.utils.data as Data
from model.resnet_light import ResNetAtt
from model.hgnet import HGNet
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.loss_fn import cross_entropy, LossWrapper, Callbacks, Callbacks2Head, cross_entropy_2head
from model.train_new import Trainer


class ConfigDecoder(object):
    def __init__(self, default: str, *args, **kwargs):
        self.yaml = YAML()
        if os.path.exists(default):
            _, ext = os.path.splitext(default)
            assert ext in [".yml", ".yaml"], "only support yaml files for now"
            self.config = self.yaml.load(open(default, "r", encoding='utf-8'))
        else:
            ValueError("Config file is not exist")

        self.merge_config(opts=kwargs)

    def merge_config(self, opts):
        for key, value in opts.items():
            if "." not in key:
                if isinstance(value, dict) and key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                sub_keys = key.split(".")
                assert sub_keys[0] in self.config, (
                    "the sub_keys can only be one of global_config: {}, but get: "
                    "{}, please check your running command".format(
                        self.config.keys(), sub_keys[0]
                    )
                )
                cur = self.config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        cur[sub_key] = value
                    else:
                        cur = cur[sub_key]

    def save_config(self, model_path, filename, save_dir=None):
        save_dir = save_dir if save_dir is not None else os.path.dirname(model_path)
        save_path = os.path.join(save_dir, filename)
        print(f"Save config to {save_path} ...")
        with open(save_path, 'w') as f:
            self.yaml.dump(self.config, f)

    def build_dataset(self):
        """
        datareader:
            reader:
                name: DatasetEditor
                dataset_path: 'datasets/cifar10'
            call:
                name: read_cifar
                # **other_params
        """
        print('reading dataset ...')
        config = self.config['datareader'].copy()
        reader_config = config.pop('reader')
        reader = eval(reader_config.pop('name'))(**reader_config)
        call_config = config.pop('call', None)
        if call_config is None:
            return reader()
        else:
            call = getattr(reader, call_config.pop('name'))
            return call(**call_config)

    def build_dataloader(self, data, mode='train'):
        """
        dataset:
            name: BasicDataset
            input_shape: (224, 224)
            random: 'strong' # if SSLDataset ['strong', ...]
        loader:
            batch_size: 64
            step_per_epoch: 250
            num_workers: 4
            shuffle: true
            pin_memory: True
            persistent_workers: True
            prefetch_factor: 2
        """
        print('building data loader: {}, ...'.format(mode))
        config = self.config[mode]['dataset'].copy()
        dataset = eval(config.pop('name'))(data=data, **config)

        config = self.config[mode]['loader'].copy()
        if 'step_per_epoch' in config and isinstance(config['step_per_epoch'], int):
            num_samples = config['batch_size'] * config.pop('step_per_epoch')
            config['sampler'] = Data.RandomSampler(data_source=dataset, num_samples=num_samples, replacement=True)
            config['shuffle'] = False
        data_loader = Data.DataLoader(dataset=dataset, **config)
        return data_loader

    def build_model(self):
        """
        model:
            pretrained_model:
                path: r'...'
                pop: ['head.attn.0.weight', 'head.attn.0.bias']
            name: ResNetAtt
            input_shape: (256, 512)
            num_classes: 10
            encoder_param: {
                'block_name': ['ResPreNorm', 'ConvSeqLight'],
                'num_block': (2, 2, 3), 'channels': (32, 64, 128),
                'first_stride': 2, 'expansion': 2, 'dropout': 0.
            },
            decoder_param: {
                'block_name': ['LinearResPreNorm', 'SpaceGate', 'CTLinearNeck'],
                'num_block': 7, 'convert_stride': 2, 'channel': 512,
                'hidden_channel': 2048, 'num_head': 4, 'dropout': 0.,
            },
            head: 'AttnHead',
        """
        print('building model ...')
        config = self.config['model'].copy()
        pretrained_config = config.pop('pretrained_model', None)
        model = eval(config.pop('name'))(**config)
        if pretrained_config is not None:
            pretrained_config = config['pretrained_model']
            weights = torch.load(pretrained_config['path'])
            if isinstance(pretrained_config['pop'], list):
                weight_popped = pretrained_config['pop']
            elif isinstance(pretrained_config['pop'], str):
                weight_popped = [pretrained_config['pop']]
            else:
                weight_popped = []
            for key in weight_popped:
                weights.pop(key)
            model.load_state_dict(weights, strict=False)
        return model

    def build_optimizer(self, model):
        """
        optimizer:
            name: Adam
            lr: 1e-4
            betas: (0.9, 0.999)
            eps:1e-8,
            weight_decay: 1e-4
            lr_scheduler:
                name: CosineAnnealingLR
                T_max: 50000
                ...
        """
        print('building optimizer ...')
        config = self.config['optimizer'].copy()
        scheduler_config = config.pop('lr_scheduler', None).copy()
        optimizer = eval(config.pop('name'))(params=filter(lambda p: p.requires_grad, model.parameters()), **config)
        if scheduler_config is not None:
            scheduler = eval(scheduler_config.pop('name'))(optimizer=optimizer, **scheduler_config)
        else:
            T_max = self.config['train']['loader']['batch_size'] * self.config['train']['loader']['step_per_epoch']
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        return optimizer, scheduler

    def build_loss(self):
        """
        loss:
            name: cross_entropy
            ...
        """
        print('building loss ...')
        config = self.config['loss'].copy()
        loss = eval(config.pop('name'))
        if isinstance(loss, types.FunctionType):
            loss = LossWrapper(loss, **config)
        elif isinstance(loss, type):
            loss = loss(**config)
        return loss

    def build_trainer(self):
        """
        program:
            name: Trainer # TrainerSimCLR
            stop_patience=15,
            monite_value: ‘loss’
            compare_type: 'min'
            log_dir=None,
            log_items: 'scalar'  # ['scalar', ...]
            verbose=True,
            callbacks: ['precise', 'recall', 'mAP']
        """
        dataset = self.build_dataset()
        loader = {}
        for key in self.config.keys():
            if any(substring in key for substring in ['train', 'valid', 'eval']):
                loader[key] = self.build_dataloader(dataset[key], mode=key)
        model = self.build_model()
        loss = self.build_loss()
        optimizer, scheduler = self.build_optimizer(model)
        print('building trainer ...')
        config = self.config['program'].copy()
        trainer = eval(config.pop('name'))
        callback_config = config.pop('callbacks')
        callbacks = eval(callback_config.pop('name', 'Callbacks'))(**callback_config)
        trainer = trainer(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            callbacks=callbacks,
            **config
        )
        self.save_config(trainer.model_path, filename='config.yml')
        return trainer

    def dataset_check(self, dataset):
        for key, value in dataset.items():
            print(key)
            if isinstance(value, dict):
                self.dataset_check(value)
            else:
                print(len(value))


def model_check():
    param = {
        'default': r'configs/water-pipline-hgnet.yml',
    }
    decoder = ConfigDecoder(**param)
    model = decoder.build_model()
    from torchinfo import summary
    input_shape = [1, 3] + decoder.config['model']['input_shape']
    summary(model, input_size=input_shape, depth=5)


def train():
    param = {
        'default': r'configs/water-pipline-hgnet.yml',
        'PS': 'try hgnet the official version: S, new attn head with bias in stem'
        # ...
    }
    decoder = ConfigDecoder(**param)
    trainer = decoder.build_trainer()
    trainer.run()


if __name__ == '__main__':
    # model_check()
    train()