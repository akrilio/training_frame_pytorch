import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config_decoder import *


class PLConfigDecoder(ConfigDecoder):

    def build_datamodule(self):
        return BaseDataModule(self)

    def build_lightning_module(self):
        return BaseLitModel(self)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: PLConfigDecoder):
        super().__init__()
        self.config = config
        self.dataset = None

    def prepare_data(self):
        self.config.build_dataset()

    def setup(self, stage=None):
        dataset = self.config.build_dataset()
        self.dataset = {
            'train': dataset.get('train'),
            'val': dataset.get('valid'),
            'test': dataset.get('test')
        }

    def train_dataloader(self):
        return self.config.build_dataloader(self.dataset['train'], mode='train')

    def val_dataloader(self):
        return self.config.build_dataloader(self.dataset['val'], mode='valid')

    def test_dataloader(self):
        return self.config.build_dataloader(self.dataset['test'], mode='test')


class BaseLitModel(pl.LightningModule):
    def __init__(self, config: PLConfigDecoder):
        super().__init__()
        self.config = config
        self.model = config.build_model()
        self.loss = config.build_loss()

        # 初始化TorchMetrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_precision = Precision(num_classes=config.config['model']['num_classes'])
        self.val_recall = Recall(num_classes=config.config['model']['num_classes'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer, scheduler = self.config.build_optimizer(self.model)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 根据原始配置调整
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        prob = y_hat.softmax(dim=-1)
        self.val_acc(prob, y)
        self.val_precision(prob, y)
        self.val_recall(prob, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        prob = y_hat.softmax(dim=-1)
        self.val_acc(prob, y)
        self.val_precision(prob, y)
        self.val_recall(prob, y)

        self.log("test_loss", loss, prog_bar=True)
        pass


class ConfigSaveCallback(pl.Callback):
    def __init__(self, config: ConfigDecoder):
        self.config = config

    def on_train_start(self, trainer, pl_module):
        self.config.save_config(
            model_path=trainer.logger.log_dir,
            filename='config.yml'
        )


def run_training(default: str):
    config = PLConfigDecoder(default)
    datamodule = config.build_datamodule()

    model = config.build_lightning_module()
    logger = TensorBoardLogger(
        save_dir=config.config['program'].get('log_dir', 'logs'),
        name=config.config['model']['name']
    )

    callbacks = [
        ConfigSaveCallback(config),
        EarlyStopping(
            monitor=config.config['program']['monite_value'],
            patience=config.config['program']['stop_patience'],
            mode=config.config['program']['compare_type']
        ),
        ModelCheckpoint(
            dirpath=logger.log_dir,
            monitor=config.config['program']['monite_value'],
            save_top_k=1,
            mode=config.config['program']['compare_type']
        )
    ]

    trainer = pl.Trainer(
        max_epochs=config.config['program']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        enable_progress_bar=config.config['program']['verbose']
    )

    trainer.fit(model, datamodule=datamodule)
    # 测试（可选）
    # trainer.test(datamodule=datamodule)
