import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary

bp_specs_dic = {
    "T": [[2, 128]],
    "S": [[1, 16], [3, 32], [1, 16]],
    "M": [[1, 64], [2, 128], [3, 256], [2, 128], [1, 64]],
}

cnn_specs_dic = {
    "M": [[1, 64], [2, 128], [3, 256], [2, 128], [1, 64]],
}


class BP_mine(nn.Module):
    def __init__(self, in_channels, out_channels, model_type=None, bias=True):
        super(BP_mine, self).__init__()

        self.model_type = model_type
        self.layer_dic = bp_specs_dic[self.model_type]
        channels = []
        for i in self.layer_dic:
            channels += i[0] * [i[1]]
        channels = np.array(channels)
        self.in_channels = np.concatenate([[in_channels], channels], axis=0)
        self.out_channels = np.concatenate([channels, [out_channels]], axis=0)
        self.active_layer = nn.GELU()  # nn.Tanh()
        self.bias = bias
        self.bp_seq = self.create_model()

    def create_model(self):
        seq = nn.Sequential()
        for n, (in_chn, out_chn) in enumerate(zip(self.in_channels, self.out_channels)):
            seq.add_module(name='layer_' + str(n),
                           module=nn.Linear(in_features=in_chn, out_features=out_chn, bias=self.bias))
            # seq.add_module(name='norm_' + str(n), module=nn.BatchNorm1d(num_features=out_chn))
            seq.add_module(name='active_' + str(n), module=self.active_layer)
            # seq.add_module(name='dropout_' + str(n), module=nn.Dropout(p=0.2))
        return seq.to(device)

    def forward(self, x):
        return self.bp_seq(x)


class Conv1DRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride, padding, bias=True):
        super(Conv1DRelu, self).__init__()

        self.cb_seq = nn.Sequential(
            nn.Conv1d(int(in_channels), int(out_channels),
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.cb_seq(inputs)
        return outputs


class Conv1DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1):
        super(Conv1DBatchNormRelu, self).__init__()

        self.cbr_seq = nn.Sequential(
            nn.Conv1d(int(in_channels), int(out_channels),
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cbr_seq(inputs)
        return outputs


class RU(nn.Module):
    """Residual Unit"""
    def __init__(self, in_chn, hid_chn, kernel_size=3, stride=1, bias=True):
        super(RU, self).__init__()

        self.conv1 = Conv1DRelu(in_chn, hid_chn, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.conv2 = Conv1DBatchNormRelu(hid_chn, in_chn, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming


class CNN_1D_mine(nn.Module):
    def __init__(self, in_channels, out_channels, model_type=None):
        super(CNN_1D_mine, self).__init__()

        self.model_type = model_type
        self.layer_dic = cnn_specs_dic[self.model_type]
        channels = []
        for i in self.layer_dic:
            channels += i[0] * [i[1]]
        channels = np.array(channels)
        self.in_channels = np.concatenate([[in_channels], channels], axis=0)
        self.out_channels = np.concatenate([channels, [out_channels]], axis=0)
        self.cnn_seq = self.create_model()

    def create_model(self):
        seq = nn.Sequential()
        for n, (in_chn, out_chn) in enumerate(zip(self.in_channels, self.out_channels)):
            seq.add_module(name='conv_' + str(n),
                           module=nn.Conv1d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=True))
            seq.add_module(name='unit_' + str(n),
                           module=RU(in_chn=out_chn, hid_chn=2 * out_chn, bias=True))
        return seq.to(device)

    def forward(self, x):
        return self.cnn_seq(x)


if __name__ == '__main__':
    # model = BP_mine(in_channels=2, out_channels=1, model_type="S")
    model = CNN_1D_mine(in_channels=1, out_channels=64, model_type="M")
    """
    x = torch.randn(150, 200, 1).cuda()
    y = torch.LongTensor(np.ones((150, 64, 1))).cuda()
    import time
    start = time.time()
    pre = model(x)
    print(pre.shape)
    print(f'spend time: {np.round(time.time() - start, 4) * 1e3} ms')

    loss = nn.MSELoss()
    print(loss(pre, y))
    """
    summary(model, input_size=(150, 1, 750))
