import pickle
import torch
from typing import Any
from model.datasets import ClassifyDataset
from model.train import Trainer
from model.resnet import ResNet
from dataset_edit import Dataset_Editor
from model.loss_fn import evaluate, cross_entropy, SimLoss


def is_picklable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError, AttributeError):
        return False
    return True


def worker_init_fn(worker_id):
    torch.cuda.set_device(torch.cuda.current_device())


def callback_loss(y_true, y_pred):
    loss = cross_entropy(y_true, y_pred)
    precise, recall, mAP = evaluate(y_true, y_pred)
    return loss, precise, recall, mAP


def main():
    editor = Dataset_Editor(dataset_path=r'datasets\cifar10')
    data = editor.read_cifar()
    # 测试你的数据集
    train_dataset = ClassifyDataset(data['train'], input_shape=(32, 32), num_classes=len(data['label_names']), random=True)
    valid_dataset = ClassifyDataset(data['train'], input_shape=(32, 32), num_classes=len(data['label_names']), random=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    print(f"Dataset is picklable: {is_picklable(train_dataset)}")
    print(f"Dataset is picklable: {is_picklable(valid_dataset)}")
    param = {'block_name': 'BottleNeck', 'num_block': [3, 4, 6, 3], 'num_classes': 10}
    model = ResNet(**param)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=256, shuffle=True, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=256, shuffle=True, pin_memory=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2,
    )
    train = Trainer(model, (train_data_loader, valid_data_loader), callback_loss,
                    lr=1e-4, weight_decay=1e-6, batch_size=256, num_works=4,
                    max_step=500, reduce_patience=8, stop_patience=25,
                    log_dir=None, verbose=True, callbacks=['loss', 'precise', 'recall', 'mAP'],
                    log_scalar=True, log_param=False)
    """"""
    train.run(num_epochs=300)


if __name__ == '__main__':
    main()
