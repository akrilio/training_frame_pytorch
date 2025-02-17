import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


class Logger(object):
    def __init__(self, log_dir=None, log_items=None, verbose=True,
                 num_epochs=0, stop_patience=False,
                 monite_value=None, delta=0, compare_type='min',
                 **kwargs):
        self.model = None
        if log_dir:
            time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join(log_dir, time)
            self.writer = SummaryWriter(log_dir)
            self.log_func = {
                'scalar': self.add_scalar,
                'params': self.add_params,
                # 'images': self.add_images, dataformats='HWC', to be continue ...
            }
            self.model_path = os.path.join(log_dir, 'checkpoint.pt')
        else:
            self.writer, self.model_path, self.log_func = None, None, None

        self.verbose = verbose
        self.train_loss = {'loss': 0}
        self.valid_loss = {'loss': 0}
        self.log_items = [log_items] if isinstance(log_items, str) else log_items
        self.cur_step = 0

        self.num_epochs = num_epochs
        self.patience = stop_patience if stop_patience else num_epochs
        self.counter = 0
        self.start, self.end = None, None
        self.monite_value, self.monite_key = None, monite_value
        init_value = {'min': 1e3, 'max': -1e3}
        self.best_value = init_value[compare_type]
        init_func = {'min': lambda: self.monite_value[self.monite_key] < self.best_value - delta,
                     'max': lambda: self.monite_value[self.monite_key] > self.best_value + delta}
        self.compare_func = init_func[compare_type]

    def add_scalar(self):
        for tag, value in self.train_loss.items():
            self.writer.add_scalar("train " + tag, value, self.cur_step)
        for tag, value in self.valid_loss.items():
            self.writer.add_scalar("valid " + tag, value, self.cur_step)

    def add_params(self):
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram(tag + '/data', value.data.cpu().numpy(), self.cur_step)
            self.writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.cur_step)

    def save_model(self, filename):
        if self.model_path:
            torch.save(self.model.state_dict(), self.model_path.replace('checkpoint', filename))

    def print_loss(self,):
        sentence = ["Epoch[{}/{}], time: {:.1f}s\n", "\t[train loss] ", "\t[valid loss] "]
        s = sentence[0].format(self.cur_step + 1, self.num_epochs, (self.end - self.start).total_seconds())
        s += sentence[1]
        for target, value in self.train_loss.items():
            s += target + ": {:.4f}; ".format(value)
        s += '\n' + sentence[2]
        for target, value in self.valid_loss.items():
            s += target + ": {:.4f}; ".format(value)
        print(s)

    def early_stopping_monite(self):
        if self.compare_func() is True:
            self.best_value = self.monite_value[self.monite_key]
            self.counter = 0
            if self.model_path:
                torch.save(self.model.state_dict(), self.model_path)
        else:
            self.counter += 1
            if self.counter < self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                return True
        return False

    def log_step(self,):
        for key in self.log_items:
            self.log_func[key]()
        if self.verbose:
            self.print_loss()
        state = self.early_stopping_monite()
        return state


class Trainer(Logger):
    def __init__(self, model, loader, loss, callbacks, optimizer, scheduler, use_amp, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss = loss
        self.callbacks = callbacks
        for key in self.callbacks.out_keys:
            self.train_loss[key] = 0
            self.valid_loss[key] = 0

        if 'valid' in loader:
            self.monite_value = self.valid_loss
        else:
            self.monite_value = self.train_loss

        if use_amp:
            self.scaler = GradScaler()
            self.train_loop = self.train_loop_amp

    def unpack(self, data):
        x, y = data
        return x.to(self.device), y.to(self.device)

    def model_forward(self, x, y):
        y_pre = self.model(x)
        loss = self.loss(y, y_pre)
        return y, y_pre, loss

    def step_record(self, recoder, y_true, y_pre, loss):
        recoder['loss'] += loss.item()
        callbacks = self.callbacks(y_true, y_pre)
        for key, value in callbacks.items():
            recoder[key] += value

    def train_loop(self, clean=False):
        self.model.train()
        for keys in self.train_loss.keys():
            self.train_loss[keys] = 0
        n = 1
        for n, data in enumerate(self.loader['train'], 1):
            data = self.unpack(data)
            y, y_pre, loss = self.model_forward(*data)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                self.step_record(self.train_loss, y, y_pre, loss)
        for keys in self.train_loss.keys():
            self.train_loss[keys] /= n
        if clean:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def train_loop_amp(self, clean=False):
        self.model.train()
        for keys in self.train_loss.keys():
            self.train_loss[keys] = 0
        n = 1
        for n, data in enumerate(self.loader['train'], 1):
            data = self.unpack(data)
            with autocast():
                y, y_pre, loss = self.model_forward(*data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()  # value
            with torch.no_grad():
                self.step_record(self.train_loss, y, y_pre, loss)
        for keys in self.train_loss.keys():
            self.train_loss[keys] /= n
        if clean:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def valid_loop(self, clean=False):
        self.model.eval()  # prep model for evaluation
        for keys in self.valid_loss.keys():
            self.valid_loss[keys] = 0
        n = 1
        with torch.no_grad():
            for n, data in enumerate(self.loader['valid'], 1):
                data = self.unpack(data)
                y, y_pre, loss = self.model_forward(*data)
                self.step_record(self.valid_loss, y, y_pre, loss)
        for keys in self.valid_loss.keys():
            self.valid_loss[keys] /= n
        if clean:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def run(self, num_epochs: int = None):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        for self.cur_step in range(self.num_epochs):
            self.start = datetime.datetime.now()
            self.train_loop(clean=True)
            if 'valid' in self.loader:
                self.valid_loop(clean=True)
            self.end = datetime.datetime.now()
            stop = self.log_step()
            if stop:
                print("Early stopping")
                break
        save_name = 'last_epoch_{}_train_loss_{:.4f}_val_loss{:.4f}'\
            .format(self.cur_step + 1, self.train_loss['loss'], self.valid_loss['loss'])
        self.save_model(save_name)


class TrainerFixMatch(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert "train_lb" in self.loader, "Fix Match need 'train_lb' loader as labeled dataloader"
        assert "train_ulb" in self.loader, "Fix Match need 'train_ulb' loader as unlabeled dataloader"
        train_lb_loader = self.loader.pop('train_lb')
        train_ulb_loader = self.loader.pop('train_ul')
        self.loader['train'] = zip(train_lb_loader, train_ulb_loader)
        self.train_loss['num_pseudo_label'] = 0
        for key in self.loss.keys():
            self.train_loss[key] = 0

    def unpack(self, data):
        x, y = data
        if isinstance(x, list or tuple):
            x, y_l = x
            x_ulw, x_uls = y
            return x.to(self.device), y_l.to(self.device), x_ulw.to(self.device), x_uls.to(self.device)
        else:
            return x.to(self.device), y.to(self.device), torch.empty().to(self.device), torch.empty().to(self.device)

    def model_forward(self, x_l, y, x_ul_w, x_ul_s):
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

        self.train_loss['sup_loss'] += sup_loss.item()
        self.train_loss['unsup_loss'] += unsup_loss.item()
        self.train_loss['num_pseudo_label'] += num
        return y, y_l, loss


class TrainerSimCLR(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_forward(self, x0, x1):
        inputs = torch.cat((x0, x1))
        outputs = self.model(inputs)
        loss = self.loss(None, outputs)
        return inputs, outputs, loss
