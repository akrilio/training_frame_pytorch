import torch
import torch.nn.functional as F


class Callbacks(object):
    def __init__(self, keys: list or tuple, eps=1e-8):
        self.keys = keys
        self.out_keys = keys.copy()
        self.func_set = [self._cal_precise, self._cal_recall, self._cal_mAP]
        self.eps = eps
        self.value = {}
        self.func_index = {
            'precise': {0},
            'recall': {0, 1},
            'mAP': {0, 1, 2}
        }
        callback_index = set()
        for key in keys:
            callback_index.update(self.func_index[key])
        callback_index = sorted(callback_index)
        self.callback_func = [self.func_set[i] for i in callback_index]

    def _cal_precise(self, y_true, y_pred):
        y_pred = torch.argmax(y_pred, dim=1)
        inter = 1. * (y_true == y_pred)
        self.value['inter'] = inter
        self.value['precise'] = torch.mean(inter)

    def _cal_recall(self, y_true, y_pred):
        trans = F.one_hot(y_true, y_pred.shape[-1]).to(torch.float32)
        weight = torch.sum(trans, dim=0)
        mask = weight != 0
        self.value['recall'] = torch.mean((self.value['inter'] @ trans)[mask] / weight[mask])

    def _cal_mAP(self, *args):
        precise = self.value['precise']
        recall = self.value['recall']
        self.value['mAP'] = 2 * precise * recall / (precise + recall + self.eps)

    def __call__(self, y_true, y_pred):
        for func in self.callback_func:
            func(y_true, y_pred)
        return {key: self.value[key].item() for key in self.keys}


def evaluate(y_true, y_pred, eps=1e-8):
    if len(y_true.shape) < len(y_pred.shape):
        y_true = F.one_hot(y_true, y_pred.shape[-1]).to(torch.float32)
    y_pred = F.one_hot(torch.argmax(y_pred, dim=-1), y_pred.shape[-1]).to(torch.float32)
    # torch.softmax(y_pred, dim=1)
    inter = y_true * y_pred
    weight = torch.sum(y_true, dim=0)
    mask = weight != 0

    precise = torch.mean(torch.sum(inter, dim=1))
    recall = torch.mean(torch.sum(inter, dim=0)[mask] / weight[mask])
    mAP = 2 * precise * recall / (precise + recall + eps)
    return precise, recall, mAP

def cross_entropy_2head(y_true, y_pred, weight_0, weight_1, device='cuda'):
    weight_0 = torch.tensor(weight_0, device=device)
    weight_1 = torch.tensor(weight_1, device=device)
    loss_0 = cross_entropy(y_true[:, 0], y_pred[:, [0, 1, 2, 3, 4, 5]], weight=weight_0)
    loss_1 = cross_entropy(y_true[:, 1], y_pred[:, [6, 7, 8, 9]], weight=weight_1)
    return loss_0 + loss_1


class Callbacks2Head(Callbacks):
    def __init__(self, keys: list or tuple, eps=1e-8):
        super().__init__(keys=keys, eps=eps)
        self.out_keys = []
        for key in self.keys:
            if key == 'mAP':
                self.out_keys.append(key)
            else:
                self.out_keys.extend([key + '_0', key + '_1'])

    def basic_call(self, y_true, y_pred):
        for func in self.callback_func:
            func(y_true, y_pred)
        return {key: self.value[key].item() for key in self.keys}

    def __call__(self, y_true, y_pred):
        eval_0 = self.basic_call(y_true[:, 0], y_pred[:, [0, 1, 2, 3, 4, 5]])
        eval_1 = self.basic_call(y_true[:, 1], y_pred[:, [6, 7, 8, 9]])
        eval_result = {}
        for key in self.keys:
            if key == 'mAP':
                eval_result[key] = eval_0[key] + eval_1[key]
            else:
                eval_result[key + '_0'] = eval_0[key]
                eval_result[key + '_1'] = eval_1[key]
        return eval_result

class LossWrapper(object):
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, y_true, y_pred):
        return self.func(y_true, y_pred, **self.kwargs)


def kl_div(y_true, y_pred):
    if len(y_true.shape) < len(y_pred.shape):
        y_true = F.one_hot(y_true, y_pred.shape[-1]).to(torch.float32)
    y_true = y_true / y_true.sum(dim=1, keepdim=True)
    y_pred = F.log_softmax(y_pred, dim=1)
    kl_loss = F.kl_div(y_pred, y_true, reduction='batchmean')
    return kl_loss


def binary_entropy(y_true, y_pred):
    y_t = (y_true == 0).to(torch.float32)
    y_p = torch.softmax(y_pred, dim=-1)[:, 0]
    entropy = F.binary_cross_entropy(y_p, y_t)
    return torch.mean(entropy)


def cross_entropy(y_true, y_pred, weight=None):
    # weight = torch.sum(y_true, dim=[2, 3]) + eps
    # water_pipline_weight = torch.tensor([0.2] + y_pred.shape[-1] * [1])
    entropy = F.cross_entropy(y_pred, y_true, weight=weight)
    return torch.mean(entropy)


def combine_entropy(y_true, y_pred, w=(0.5, 0.5)):
    return w[0] * cross_entropy(y_true, y_pred) + w[1] * binary_entropy(y_true, y_pred)


def recall_cross_entropy(y_true, y_pred, eps=1e-8):
    y_pred = torch.softmax(y_pred, dim=1)
    recall = y_true * (y_pred - 1)
    entropy = torch.sum(recall * torch.log(y_pred + eps), dim=1)
    return torch.mean(entropy)


def focal_loss(y_true, y_pred, alpha=1, gamma=2):
    entropy = F.cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-entropy)
    loss = alpha * (1 - pt) ** gamma * entropy
    return torch.mean(loss)


class ConsistencyLoss(object):
    def __init__(self, b=32, T=0.5, p_cutoff=0.95):
        self.b = b
        self.T = T
        self.p_cutoff = p_cutoff
        self.eps = 1e-9

    def soft_consistency(self, y_w, y_s):
        y_w = torch.clamp(y_w, -10, 10)
        y_s = torch.clamp(y_s, -10, 10)
        p_w = torch.log_softmax(y_w / self.T, dim=1)
        p_s = torch.softmax(y_s / self.T, dim=1) + self.eps
        unmasked_loss = torch.sum(F.kl_div(p_w, p_s, reduction="none"), dim=1)
        """
        p_w = torch.softmax(y_w / self.T, dim=1)
        p_s = torch.softmax(y_s / self.T, dim=1)
        # unmasked_loss = 1 - F.cosine_similarity(p_w, p_s, dim=1)
        unmasked_loss = F.mse_loss(p_s, p_w, reduction='sum')
        """
        return unmasked_loss

    def __call__(self, y_w, y_s):
        y_w = y_w.detach()
        max_probs, max_idx = torch.max(torch.softmax(y_w, dim=1), dim=1)
        mask = max_probs.ge(self.p_cutoff).to(torch.float32)
        masked_loss = F.cross_entropy(y_s, max_idx)
        unmasked_loss = self.soft_consistency(y_w, y_s)
        return (mask * masked_loss).mean() + ((1 - mask) * unmasked_loss).mean(), mask.mean()


class SimLoss(object):
    def __init__(self, batch_size, temperature=0.07):
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = torch.eye(batch_size).bool()
        label = torch.arange(batch_size // 2).cuda()
        self.label = torch.cat([label + batch_size // 2 - 1, label], dim=0)

    def __call__(self, y_true, y_pred):
        features = F.normalize(y_pred, dim=1)
        mask_pre = (features @ features.T)[~self.mask].reshape(self.batch_size, -1)
        return F.cross_entropy(mask_pre / self.temperature, self.label)


if __name__ == '__main__':
    """
    x, y = torch.randn(size=(2, 4, 4))
    loss_fn = ConsistencyLoss(p_cutoff=0.8)
    print(loss_fn.soft_consistency(x, y))
    """
    y_true = torch.randint(low=0, high=4, size=(9, 2))
    y_pred = torch.randn(size=(9, 9))
    loss_fn = evaluate(y_true, y_pred)
    print(loss_fn)
