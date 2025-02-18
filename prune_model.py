import os
import glob
import types
import torch
import torch.nn as nn
import torch.ao.quantization as quantization
import torch.utils.data as Data
import torch_pruning as tp
import onnxruntime as ort

from model.train import TrainerMixPrecise
from model.datasets import BasicDataset
from model.loss_fn import cross_entropy, evaluate
from image_searcher import Searcher


class QuantNet(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = original_model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class Pruner(object):
    def __init__(self, file_path=None, image_shape=(256, 512)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_path = file_path
        self.image_shape = image_shape
        self.model = None
        self.weights_path = None
        self.data_loader = None
        self.callbacks = {
            'func': evaluate,
            'keys': ['precise_0', 'recall_0', 'mAP', 'precise_1', 'recall_1', 'mAP']  # ['precise', 'recall', 'mAP']
        }
        self.step_per_epoch = 100

    def load_model(self, model_path, load_type='base'):
        if load_type == 'base':
            self.model = torch.load(model_path).to(self.device)
        elif load_type == 'jit':
            self.model = torch.jit.load(model_path).to(self.device)
        elif load_type == 'quant':
            assert load_type in model_path
            self.model = torch.jit.load(model_path, map_location='cpu')

    def read_dataset(self, dataset_path, key='缺陷', step_per_epoch=250, batch_size=16):
        torch.multiprocessing.set_start_method('spawn', force=True)
        print('loading dataset ...')
        searcher = Searcher(root_path=None, dataset_path=dataset_path)
        lb_dataset, _ = searcher.read_dataset(label=key, equal=False, T=1)

        train_dataset = BasicDataset(lb_dataset['train'], self.image_shape, 'strong')
        valid_dataset = BasicDataset(lb_dataset['valid'], self.image_shape, 'valid')

        train_loader = Data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
            num_workers=12, persistent_workers=True, prefetch_factor=2,
            sampler=Data.RandomSampler(train_dataset, num_samples=batch_size * step_per_epoch, replacement=True),
        )
        valid_loader = Data.DataLoader(
            dataset=valid_dataset, batch_size=2 * batch_size, shuffle=False, pin_memory=True,
            num_workers=1, persistent_workers=True, prefetch_factor=2,
        )
        self.step_per_epoch = step_per_epoch
        self.data_loader = (train_loader, None, valid_loader)
        self.classes = lb_dataset['label_names']

    def tp_prune(self, num=1, ratio=0.5):
        # summary(self.model.body[-1].blocks, input_size=(1, 256, 2048))
        example_inputs = torch.randn((1, 512, 512)).to(self.device)
        imp = tp.importance.GroupNormImportance()
        prune_units = []
        for blk in self.model.decoder:
            if blk.__class__.__name__ == 'LinearAttV2':
                pruned_model = blk.linear_0
                prune_unit = tp.pruner.MetaPruner(
                    pruned_model, example_inputs, importance=imp,
                    iterative_steps=num, pruning_ratio=ratio,
                    ignored_layers=[pruned_model[-3]],
                )
                prune_units.append(prune_unit)

                pruned_model = blk.linear_1
                prune_unit = tp.pruner.MetaPruner(
                    pruned_model, example_inputs, importance=imp,
                    iterative_steps=num, pruning_ratio=ratio,
                    ignored_layers=[pruned_model[-3]],
                )
                prune_units.append(prune_unit)
        return prune_units

    def fine_tune(self, epoch):
        torch.multiprocessing.set_start_method('spawn', force=True)
        print('setting trainer ...')
        train = TrainerMixPrecise(
            model=self.model, dataloader=self.data_loader, loss=cross_entropy,
            lr=1e-5, weight_decay=1e-3, reduce_patience=8, stop_patience=150,
            verbose=True, callbacks=self.callbacks, log_dir=None, log_scalar=True, log_param=False
        )
        print('running ...')
        train.run(num_epochs=epoch, step_per_epochs=self.step_per_epoch)

    def quantization(self):
        modify_module(self.model)
        model = QuantNet(self.model)
        model.eval()
        model.qconfig = quantization.get_default_qconfig('x86')  # 'qnnpack'
        model_prepared = quantization.prepare(model)
        with torch.no_grad():
            for (image_data, _) in self.data_loader[-1]:
                model_prepared(image_data.to(self.device))
        model_prepared.to('cpu')
        self.model = quantization.convert(model_prepared)

    def test_on_dataset(self, device=None):
        y_true = []
        y_pred = []
        if device not in ('cpu', 'cuda'):
            device = self.device
        self.model.eval()
        print('testing ... ')
        with torch.no_grad():
            for (image_data, label_data) in self.data_loader[0]:
                image_data = image_data.to(device)
                label_data = label_data.to(device)
                pre = self.model(image_data)
                y_true.append(label_data)
                y_pred.append(pre)
        if device == 'cuda':
            torch.cuda.empty_cache()
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        callbacks = self.callbacks['func'](y_true, y_pred)
        verbose = '[test] '
        for key, value in zip(self.callbacks['keys'], callbacks):
            verbose += "{}:{}, ".format(key, value)
        print(verbose)

    def prune_model(self, num=3, amount=0.5, epoch=50):
        pruners = self.tp_prune(num, amount)
        for i in range(num):
            print('round {}:'.format(i + 1))
            for p in pruners:
                p.step()
            self.fine_tune(epoch)
            self.test_on_dataset()
        pruned_model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Pruned model size: {pruned_model_size}")

    def save_model(self, file_name, save_type='base'):
        self.model.eval()
        if save_type == 'base':
            torch.save(self.model, file_name)
        elif save_type == 'jit':
            input_tensor = torch.randn(size=(1, 3, 256, 512), dtype=torch.float32).to(self.device)
            script = torch.jit.trace(self.model, input_tensor)
            # script = torch.jit.script(pruner.model)
            torch.jit.save(script, file_name)
        elif save_type == 'quant':
            assert save_type in file_name
            input_tensor = torch.randn(size=(1, 3, 256, 512), dtype=torch.float32, device='cpu')
            script = torch.jit.trace(self.model, input_tensor)
            torch.jit.save(script, file_name)

    def select_model(self, index, file_path=None):
        if file_path is not None:
            self.file_path = file_path
            if os.path.exists(self.file_path) is False:
                os.mkdir(file_path)
        for key, value in index.items():
            weights_path = glob.glob(os.path.join(self.file_path, 'prune-*{}-*.pt'.format(key)))
            for i, path in enumerate(weights_path):
                if i not in value:
                    os.remove(path)

    def to_onnx(self, file_path=None, path_mod='{}-prune-{}.{}'):
        if file_path is not None:
            self.file_path = file_path
            if os.path.exists(self.file_path) is False:
                os.mkdir(file_path)
        for key in ('缺陷', '管道附属设备及管配件'):
            weights_path = glob.glob(os.path.join(file_path, path_mod.format(key, '*', 'pt')))
            input_tensor = torch.randn(size=(1, 3, 256, 512), device='cpu')
            for n, path in enumerate(weights_path):
                try:
                    model = torch.load(path)
                except:
                    model = torch.jit.load(path)
                model.eval()
                onnx_name = os.path.join(file_path, path_mod.format(key, n, 'onnx'))
                print('{} to {}'.format(path, onnx_name))
                torch.onnx.export(
                    model, input_tensor, onnx_name,
                    export_params=True, opset_version=11,  do_constant_folding=True,
                    input_names=['input'], output_names=['output'], dynamic_axes={'input': [0]},
                )

    def compare_torch_onnx(self, name_mod):
        if isinstance(name_mod, (tuple, list)):
            torch_model_name, onnx_model_name = name_mod
        elif isinstance(name_mod, str):
            torch_model_name = name_mod.format('pt')
            onnx_model_name = name_mod.format('onnx')
        torch_model = torch.jit.load(torch_model_name)
        onnx_model = ort.InferenceSession(onnx_model_name, providers=['CPUExecutionProvider'])
        print('load model: \n {} \n {}'.format(torch_model_name, onnx_model_name))
        torch_model.eval()
        test_data = torch.randn(size=(20, 3, 256, 512))
        with torch.no_grad():
            torch_result = torch_model(test_data)
            test_data = test_data.detach().numpy()
            onnx_result = onnx_model.run(None,  {"input": test_data})[0]
            onnx_result = torch.tensor(onnx_result)
        print('inference ok')
        print(torch.mean(torch_result ** 2), torch.mean((torch_result - onnx_result) ** 2))


def squeeze_0():
    named_key = {'缺陷': 'defect', '管道附属设备及管配件': 'equipment'}
    file_path = 'inference_param'
    pruner = Pruner()
    input_tensor = torch.randn(size=(1, 3, 256, 512))
    for key in ('缺陷', '管道附属设备及管配件'):
        # pruner.read_dataset(dataset_path=r'D:\Public\water_pipeline', key=key, step_per_epoch=100, batch_size=16)
        model_path = glob.glob(os.path.join(file_path, 'prune-*{}*.pt'.format(key)))
        for i, path in enumerate(model_path):
            """"""
            pruner.model = torch.load(path, map_location='cuda')
            pruner.model.eval()
            modify_module(pruner.model)
            pruner.test_on_dataset()
            pruner.prune_model(num=3, amount=0.75, epoch=10)
            pruner.quantization()
            pruner.test_on_dataset(device='cpu')
            model_name = os.path.join(file_path, '{}-prune-quantize-{}.pt'.format(named_key[key], i))
            pruner.save_model(model_name)
            print('{} to {}'.format(path, model_name))
            onnx_name = model_name.replace(named_key[key], key).replace('.pt', '.onnx')
            """
            model = torch.jit.load(model_name, map_location='cpu')
            torch.onnx.export(
                model, input_tensor, onnx_name,
                export_params=True, opset_version=16, do_constant_folding=True,
                input_names=['input'], output_names=['output'], dynamic_axes={'input': [0]},
            )
            print('{} to {}'.format(model_name, onnx_name))
            pruner.compare_torch_onnx((model_name, onnx_name))
            """
    index = {'缺陷': [3, 4], '管道附属设备及管配件': [2, 3]}
    pruner.select_model(index=index, file_path=r'inference_param')
    pruner.to_onnx(file_path=file_path)


def quantized_bottleneck_forward(self, x):
    return self.activation(self.add_func.add(self.residual_function(x), self.shortcut(x)))


def quantized_linearatt_forward(self, x):
    x = self.add_func.add(x, self.linear_0(x))
    x = x.transpose(1, 2)
    x = self.add_func.add(x, self.linear_1(x))
    x = x.transpose(1, 2)
    return x


def quantized_resnet_forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    x = x.view(x.size(0), -1)
    x = self.output(x)
    return x


def modify_module(module):
    module.forward = types.MethodType(quantized_resnet_forward, module)
    print('changed module.forward')
    for name, child in module.named_modules():
        if child.__class__.__name__ == 'BottleNeck':
            print('find BottleNeck: {}'.format(name))
            child.add_func = nn.quantized.FloatFunctional()
            child.forward = types.MethodType(quantized_bottleneck_forward, child)
            child.fuse_model = True
            quantization.fuse_modules(child,
                                      [['residual_function.0', 'residual_function.1']], inplace=True)
            quantization.fuse_modules(child,
                                      [['residual_function.4', 'residual_function.5']], inplace=True)
            quantization.fuse_modules(child,
                                      [['residual_function.8', 'residual_function.9']], inplace=True)
            if isinstance(child.shortcut, nn.Sequential):
                quantization.fuse_modules(child,
                                          [['shortcut.0', 'shortcut.1']], inplace=True)
        elif child.__class__.__name__ == 'BasicBlock':
            print('find BottleNeck: {}'.format(name))
            child.add_func = nn.quantized.FloatFunctional()
            child.forward = types.MethodType(quantized_bottleneck_forward, child)
            child.fuse_model = True
            quantization.fuse_modules(child,
                                      [['residual_function.0', 'residual_function.1']], inplace=True)
            quantization.fuse_modules(child,
                                      [['residual_function.4', 'residual_function.5']], inplace=True)
            if isinstance(child.shortcut, nn.Sequential):
                quantization.fuse_modules(child,
                                          [['shortcut.0', 'shortcut.1']], inplace=True)
        elif child.__class__.__name__ in ('LinearAtt', 'LinearAttV2'):
            print('find LinearAtt: {}'.format(name))
            child.add_func = nn.quantized.FloatFunctional()
            child.forward = types.MethodType(quantized_linearatt_forward, child)
        elif child.__class__.__name__ == 'LeakyReLU':
            print('find LeakyReLU: {}'.format(name))
            child.inplace = False
        elif name in ('body.0', 'encoder.0'):
            child.fuse_model = True
            quantization.fuse_modules(child, [['0', '1']], inplace=True)
            print('find input: {}'.format(name))
    return module


from torch.profiler import profile, record_function, ProfilerActivity


def profile_model(model, input_data, model_name):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model.eval()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_data)
    print(f"\nProfiling results for {model_name}:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(f"{model_name}_trace.json")


if __name__ == '__main__':
    file_path = 'inference_param'
    pruner = Pruner()
    pruner.read_dataset(dataset_path=r'D:\Public\water_pipeline', key='all', step_per_epoch=100, batch_size=128)
    model_path = glob.glob(os.path.join(file_path, 'all-2024*.pt'))
    for i, path in enumerate(model_path):
        pruner.load_model(path, load_type='base')
        pruner.test_on_dataset()
        # pruner.prune_model(num=5, amount=0.5, epoch=10)
        # pruner.quantization()
        # pruner.test_on_dataset(device='cpu')
        # model_name = path.replace('.pt', '.onnx')
        # pruner.save_model(model_name, save_type='quant')
        """
        input_tensor = torch.randn(size=(1, 3, 256, 512))
        torch.onnx.export(
            pruner.model, input_tensor, model_name,
            export_params=True, opset_version=16, do_constant_folding=True,
            input_names=['input'], output_names=['output'], dynamic_axes={'input': [0]},
        )

        print('{} to {}'.format(path, model_name))
        """
    """
    prune-2_缺陷-20240922-165653-last_epoch_132_train_loss_0.7222_val_loss1.3978.pt to 缺陷-prune-0.onnx
    prune-quantize_缺陷-20240919-081432-last_epoch_200_train_loss_0.7099_val_loss1.4802.pt to 缺陷-prune-1.onnx
    prune-2_管道附属设备及管配件-20240922-124740-last_epoch_168_train_loss_0.3231_val_loss0.9120.pt to 管道附属设备及管配件-prune-0.onnx
    prune-2_管道附属设备及管配件-20240922-194207-last_epoch_138_train_loss_0.4110_val_loss1.2428.pt to 管道附属设备及管配件-prune-1.onnx
    """
