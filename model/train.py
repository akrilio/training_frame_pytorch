import os
import datetime
import torch
import json
from torch.cuda.amp import autocast, GradScaler

from .pytorchtools import Logger, EarlyStopping


class Trainer(object):
    def __init__(self, model, dataloader, loss, lr, weight_decay=1e-4, reduce_patience=5, stop_patience=15,
                 log_dir=None, verbose=True, callbacks=None, log_scalar=False, log_param=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_data_loader, _, self.valid_data_loader = dataloader
        self.loss = loss
        self.callbacks = callbacks
        self.train_loss = {'loss': 0}
        if callbacks is not None:
            for key in self.callbacks['keys']:
                self.train_loss[key] = 0
        self.valid_loss = self.train_loss.copy()

        self.log_dir = log_dir
        self.log_scalar = log_scalar
        self.log_param = log_param
        self.verbose = verbose

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=lr, weight_decay=weight_decay)
        """
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=reduce_patience)
        
        """

        self.scheduler = None

        self.create_path(stop_patience)

    def create_path(self, stop_patience):
        if self.log_dir is not None:
            time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            self.log_dir = os.path.join(self.log_dir, "{}-{}".format(self.model.name, time))
            self.model_path = os.path.join(self.log_dir, 'checkpoint.pt')
            self.logger = Logger(self.log_dir)
            self.earlystopping = EarlyStopping(patience=stop_patience, path=self.model_path, verbose=False)
            with open(os.path.join(self.log_dir, "model_params.json"), 'w') as write_f:
                write_f.write(json.dumps(self.model.params, indent=4, ensure_ascii=False))
            write_f.close()
        else:
            self.earlystopping = EarlyStopping(patience=stop_patience, path=None, verbose=False)

    def create_log(self, step, log_scalar=False, log_parameters=False):
        # 1. Log scalar values (scalar summary)
        if log_scalar:
            for tag, value in self.train_loss.items():
                self.logger.scalar_summary("train " + tag, value, step)
            for tag, value in self.valid_loss.items():
                self.logger.scalar_summary("valid " + tag, value, step)
        # 2. Log values and gradients of the parameters (histogram summary)
        if log_parameters:
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag + '/data', value.data.cpu().numpy(), step)
                self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

    def unpack(self, data):
        x, y = data
        return x.to(self.device), y.to(self.device)

    def _train_loop(self):
        n = 1
        for n, data in enumerate(self.train_data_loader, 1):
            x, y = self.unpack(data)
            # Forward pass
            y_pre = self.model(x)
            loss = self.loss(y, y_pre)
            # Backprop and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # value
            # record training loss
            self.train_loss['loss'] += loss.item()
            if self.callbacks is not None:
                with torch.no_grad():
                    callbacks = self.callbacks['func'](y, y_pre)
                    for i, key in enumerate(self.callbacks['keys']):
                        self.train_loss[key] += callbacks[i].item()
        return n

    def update_param(self, clean=False):
        self.model.train()
        for keys in self.train_loss.keys():
            self.train_loss[keys] = 0
        n = self._train_loop()
        for keys in self.train_loss.keys():
            self.train_loss[keys] /= n
        if clean:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _valid_loop(self):
        n = 1
        with torch.no_grad():
            for n, data in enumerate(self.valid_data_loader, 1):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                y_pre = self.model(x)
                loss = self.loss(y, y_pre)
                self.valid_loss['loss'] += loss.item()
                if self.callbacks is not None:
                    callbacks = self.callbacks['func'](y, y_pre)
                    for i, key in enumerate(self.callbacks['keys']):
                        self.valid_loss[key] += callbacks[i].item()
        return n

    def cal_valid(self, clean=False):
        self.model.eval()  # prep model for evaluation
        for keys in self.valid_loss.keys():
            self.valid_loss[keys] = 0
        n = self._valid_loop()
        for keys in self.valid_loss.keys():
            self.valid_loss[keys] /= n
        if clean:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def monitor(self, value):
        self.earlystopping(value, self.model)

    def print_loss(self, num_epochs, epoch, start, end):
        sentence = ["Epoch[{}/{}], time: {:.1f}s\n", "\t[train] ", "\t[valid] "]
        s = sentence[0].format(epoch + 1, num_epochs, (end - start).total_seconds())
        s += sentence[1]
        for target, value in self.train_loss.items():
            s += target + ": {:.4f}; ".format(value)
        s += '\n' + sentence[2]
        for target, value in self.valid_loss.items():
            s += target + ": {:.4f}; ".format(value)
        print(s)

    """
    def cleanup(self):
        if self.num_works > 0 and hasattr(self, 'train_data_loader'):
            self.train_data_loader._iterator.clean()
    """

    def run(self, num_epochs, step_per_epochs=500):
        epoch = 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs * step_per_epochs)
        # self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_epochs * step_per_epochs, num_warmup_steps=0)
        for epoch in range(num_epochs):
            start = datetime.datetime.now()
            self.update_param(clean=True)
            if self.valid_data_loader is not None:
                self.cal_valid(clean=True)
            end = datetime.datetime.now()
            if self.log_dir is not None:
                self.create_log(epoch, log_scalar=self.log_scalar, log_parameters=self.log_param)
            if self.verbose:
                self.print_loss(num_epochs, epoch, start, end)
            if self.valid_data_loader is not None:
                self.monitor(-self.valid_loss['mAP'])
            else:
                self.monitor(self.train_loss['loss'])
            if self.earlystopping.early_stop:
                print("Early stopping")
                break
        if self.log_dir:
            saved_name = 'last_epoch_{}_train_loss_{:.4f}_val_loss{:.4f}'\
                .format(epoch + 1, list(self.train_loss.values())[0], list(self.valid_loss.values())[0])
            last_epoch_path = self.model_path.replace('checkpoint', saved_name)
            torch.save(self.model.state_dict(), last_epoch_path)


class TrainerMixPrecise(Trainer):
    def __init__(self, **params):
        super().__init__(**params)
        self.scaler = GradScaler()

    def _train_loop(self):
        n = 1
        for n, data in enumerate(self.train_data_loader, 1):
            x, y = self.unpack(data)
            # Forward pass
            with autocast():
                y_pre = self.model(x)
                loss = self.loss(y, y_pre)
            # Backprop and optimize
            self.scaler.scale(loss).backward()
            # if n % 8 == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()  # value
            # record training loss
            self.train_loss['loss'] += loss.item()
            if self.callbacks is not None:
                with torch.no_grad():
                    callbacks = self.callbacks['func'](y, y_pre)
                    for i, key in enumerate(self.callbacks['keys']):
                        self.train_loss[key] += callbacks[i].item()
        return n


class TrainerFixMatch(Trainer):
    def __init__(self, **params):
        super().__init__(**params)
        self.train_l_loader, self.train_ul_loader, self.valid_data_loader = params['dataloader']
        self.train_loss = {'loss': 0, 'num_pseudo_label': 0}
        self.valid_loss = {'loss': 0}
        for key in self.loss.keys():
            self.train_loss[key] = 0
        if self.callbacks is not None:
            for key in self.callbacks['keys']:
                self.valid_loss[key] = 0

    def _train_loop(self):
        n = 1
        for n, (data_l, data_ul) in enumerate(zip(self.train_l_loader, self.train_ul_loader), 1):
            x_l, y = self.unpack(data_l)
            x_ul_w, x_ul_s = self.unpack(data_ul)
            num_l = x_l.shape[0]

            inputs = torch.cat((x_l, x_ul_s))
            outputs = self.model(inputs)
            y_l, y_ul_s = outputs[:num_l], outputs[num_l:]
            del inputs, outputs
            with torch.no_grad():
                y_ul_w = self.model(x_ul_w)
            """
            inputs = torch.cat((x_l, x_ul_w, x_ul_s))
            outputs = self.model(inputs)
            y_l = outputs[:num_l]
            y_ul_w, y_ul_s = outputs[num_l:].chunk(2)
            del inputs, outputs
            """
            sup_loss = self.loss['sup_loss'](y, y_l)
            unsup_loss, num = self.loss['unsup_loss'](y_ul_w, y_ul_s)
            loss = sup_loss + unsup_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # value

            self.train_loss['loss'] += loss.item()
            self.train_loss['sup_loss'] += sup_loss.item()
            self.train_loss['unsup_loss'] += unsup_loss.item()
            self.train_loss['num_pseudo_label'] += num
        return n

    def _valid_loop(self):
        n = 1
        with torch.no_grad():
            for n, data in enumerate(self.valid_data_loader, 1):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                y_pre = self.model(x)
                loss = self.loss['sup_loss'](y, y_pre)
                callbacks = self.callbacks['func'](y, y_pre)
                self.valid_loss['loss'] += loss.item()
                for i, key in enumerate(self.callbacks['keys']):
                    self.valid_loss[key] += callbacks[i].item()
        return n


class TrainerSimCLR(Trainer):
    def __init__(self, **params):
        super().__init__(**params)
        self.scaler = GradScaler()
        self.valid_loss['mAP'] = 0

    def _train_loop(self):
        n = 1
        for n, data in enumerate(self.train_data_loader, 1):
            x0, x1 = self.unpack(data)
            inputs = torch.cat((x0, x1))
            with autocast():
                outputs = self.model(inputs)
                loss = self.loss(None, outputs)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.train_loss['loss'] += loss.item()
        return n

    def _valid_loop(self):
        n = 1
        with torch.no_grad():
            for n, data in enumerate(self.valid_data_loader, 1):
                x0, x1 = self.unpack(data)
                inputs = torch.cat((x0, x1))
                outputs = self.model(inputs)
                loss = self.loss(None, outputs)
                self.valid_loss['loss'] += loss.item()
                self.valid_loss['mAP'] += -loss.item()
        return n


"""
if __name__ == '__main__':
    x = torch.linspace(1, 11, 11).to(torch.float32)
    y = torch.linspace(11, 1, 11).to(torch.float32)
    torch_dataset = Data.TensorDataset(x, y)

    model = BP_mine(in_channels=2, out_channels=1, model_type="M")
    train = Trainer(model, torch_dataset, torch.nn.MSELoss(), lr=1e-4,
                    log_dir=r'logs', verbose=True, log_scalar=False, log_param=False)
    train.run(num_epochs=100)
"""
