"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch
import torch.nn as nn
from torchinfo import summary
from einops.layers.torch import Rearrange


class Conv2dNorm(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.weight.flatten(1), dim=-1)
        weight = weight.reshape((self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        return self._conv_forward(input, weight, self.bias)


class SEBlock(nn.Module):
    def __init__(self, in_channel, expansion=0.25):
        super(SEBlock, self).__init__()
        self.in_channel = in_channel
        hidden_channel = int(in_channel * expansion)
        self.attn = nn.Sequential(
            nn.Linear(in_channel, hidden_channel, bias=False),
            nn.Hardswish(inplace=True),
            nn.Linear(hidden_channel, in_channel, bias=False),
            nn.Sigmoid(),
        )


class SEBlock1d(SEBlock):
    def __init__(self, in_channel, expansion=0.25):
        super().__init__(in_channel, expansion)
        self.squeeze = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        s = self.squeeze(x.permute((0, 2, 1)))
        s = s.view(-1, self.in_channel)
        s = self.attn(s)
        s = s.view(-1, 1, self.in_channel)
        return x * s


class SEBlock2d(SEBlock):
    def __init__(self, in_channel, expansion=0.25):
        super().__init__(in_channel, expansion)
        self.squeeze = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        s = self.squeeze(x)
        s = s.view(-1, self.in_channel)
        s = self.attn(s)
        s = s.view(-1, self.in_channel, 1, 1)
        return x * s


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """

    def __init__(self, in_channels, out_channels, stride=1, expansion=1, dropout=0.1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion, momentum=0.001),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.001)
        )
        # shortcut
        self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.001)
            )

    def forward(self, x):
        return self.activation(self.residual_function(x) + self.shortcut(x))


class BasicBlockPreNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, dropout=0.1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=0.001),
            nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1, bias=False),
        )
        # shortcut
        # self.activation = nn.Hardswish(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class BasicBlockSaPreNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, dropout=0.1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=0.001),
            nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels * expansion, out_channels + 1, kernel_size=3, padding=1, bias=False),
        )
        self.gate = nn.Sigmoid()
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)

    def forward(self, x):
        y = self.residual_function(x)
        y[:, 1:] *= self.gate(y[:, :1])
        return self.shortcut(x) + y[:, 1:]


class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, dropout=0.1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.001),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),  # nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.001),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion, momentum=0.001),
        )

        self.shortcut = nn.Identity()
        self.activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        if stride != 1 or in_channels != out_channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, stride=stride, kernel_size=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion, momentum=0.001)
            )

    def forward(self, x):
        return self.activation(self.residual_function(x) + self.shortcut(x))


class LinearAtt(nn.Module):
    def __init__(self, input_shape, inner_shape, dropout=0.):
        super().__init__()
        n, e = input_shape
        n_in, e_in = inner_shape
        # self.activation = nn.Hardswish(inplace=True)
        self.linear_0 = nn.Sequential(
            nn.LayerNorm(e),
            nn.Linear(e, e_in),
            nn.GELU(),
            nn.Linear(e_in, e),
        )
        self.linear_1 = nn.Sequential(
            nn.LayerNorm(n),
            nn.Linear(n, n_in),
            nn.GELU(),
            nn.Linear(n_in, n),
        )

    def forward(self, x):
        x = x + self.linear_0(x)
        x = x.transpose(1, 2)
        x = x + self.linear_1(x)
        x = x.transpose(1, 2)
        return x


class LinearSa(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm(in_channel),
            nn.Linear(in_channel, hidden_channel),
            nn.GELU(),
            nn.Linear(hidden_channel, out_channel + 1)
        )
        self.gate = nn.Sigmoid()
        self.out_channel = out_channel

    def forward(self, x):
        y = self.linear(x)
        return y[..., :self.out_channel] * self.gate(y[..., self.out_channel:])


class LinearAttSa(nn.Module):
    def __init__(self, input_shape, inner_shape, dropout=0.):
        super().__init__()
        n, e = input_shape
        n_in, e_in = inner_shape
        self.linear_0 = LinearSa(e, e_in, e)
        self.linear_1 = LinearSa(n, n_in, n)

    def forward(self, x):
        x = x + self.linear_0(x)
        x = x.transpose(1, 2)
        x = x + self.linear_1(x)
        x = x.transpose(1, 2)
        return x


class LinearAttV2(nn.Module):
    def __init__(self, input_shape, inner_shape, dropout=0.):
        super().__init__()
        n, e = input_shape
        n_in, e_in = inner_shape
        self.linear_0 = nn.Sequential(
            nn.LayerNorm(e),
            nn.Linear(e, e_in),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(e_in, e),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.linear_1 = nn.Sequential(
            nn.LayerNorm(n),
            nn.Linear(n, n_in),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_in, n),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.linear_0(x)
        x = x.transpose(1, 2)
        x = x + self.linear_1(x)
        x = x.transpose(1, 2)
        return x


class AttnHead(nn.Module):
    def __init__(self, in_channel, num_class):
        super().__init__()
        self.num_class = num_class
        self.attn = nn.Sequential(
            nn.Linear(in_channel, num_class),
            nn.Softmax(dim=1)
        )
        self.proj = nn.Linear(in_channel, 1)

    def forward(self, x):
        a = self.attn(x)
        x = a.transpose(1, 2) @ x
        x = self.proj(x)
        return x.view(-1, self.num_class)


class ResNetAtt(nn.Module):
    def __init__(self, input_shape, encoder_param, decoder_param, num_classes):
        super().__init__()
        """
        param = {'input_shape': (256, 512), 'num_classes': 10,
                 'encoder_param': {
                     'block_name': 'BottleNeck', 'num_block': (3, 4, 6, 3), 
                     'channels': (32, 64, 128, 256), 'first_stride': 1, 
                     'expansion': 2, 'dropout': 0.1
                 },
                 'decoder_param':{
                     'block_name': 'LinearAttV3', 'inner_shape': (64, 64),
                     'num_block': 4, 'channels': 512, 'convert_stride': 2,
                     'dropout': 0.1,
                 }
                }
        """
        self.name = "ResNetAtt"
        self.params = {'input_shape': input_shape, 'num_classes': num_classes,
                       'encoder_param': encoder_param, 'decoder_param': decoder_param}
        self.input_shape = [input_shape[0], input_shape[1]]
        self.in_channels = 3
        self.encoder = self._make_resnet_encoder(**encoder_param)
        self.decoder = self._make_attn_decoder(**decoder_param)
        self.output = AttnHead(self.in_channels, num_classes)
        # nn.Linear(self.in_channels, num_classes)

    def _make_resnet_encoder(self, block_name, num_block, channels, expansion, first_stride, dropout=0.):
        block = eval(block_name)
        stride = [1] + [2] * (len(channels) - 1)
        layers = [
            nn.Conv2d(self.in_channels, channels[0], kernel_size=first_stride, stride=first_stride),
            nn.BatchNorm2d(channels[0])
        ]
        self.in_channels = channels[0]
        self.input_shape = [i // first_stride for i in self.input_shape]
        for ch, n, s in zip(channels, num_block, stride):
            layers.append(self._make_resnet_block(block, ch, n, s, expansion, dropout))
            self.input_shape = [i // s for i in self.input_shape]
        return nn.Sequential(*layers)

    def _make_resnet_block(self, block, out_channels, num_blocks, stride, expansion, dropout):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, expansion, dropout))
            self.in_channels = out_channels
            if block.__name__ == 'BottleNeck':
                self.in_channels *= expansion
        return nn.Sequential(*layers)

    def _make_attn_decoder(self, block_name, num_block, channels, inner_shape, convert_stride, dropout=0.):
        block = eval(block_name)
        self.input_shape = [i // convert_stride for i in self.input_shape]
        h, w = self.input_shape
        convert = nn.Sequential(
            nn.Conv2d(self.in_channels, channels, kernel_size=convert_stride, stride=convert_stride),
            Rearrange('b n h w -> b (h w) n'),
        )
        self.in_channels = channels
        input_shape = (channels, h * w)
        layers = [convert]
        for _ in range(num_block):
            layers.append(block(input_shape, inner_shape, dropout))
        # layers.append(nn.Linear(h * w, 1), nn.Flatten(1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x


""" 
ResNet 18 object
    param = {'block_name': 'BasicBlock',
             'num_block': [2, 2, 2, 2], 'channels': [64, 128, 256, 512],
             'num_classes': 100, 'expansion': 1}
ResNet 34 object
    param = {'block_name': 'BasicBlock',
             'num_block': [3, 4, 6, 3], 'channels': [64, 128, 256, 512],
             'num_classes': 100, 'expansion': 1}
ResNet 50 object
    param = {'block_name': 'BottleNeck',
             'num_block': [3, 4, 6, 3], 'channels': [64, 128, 256, 512],
             'num_classes': 100, 'expansion': 4}
ResNet 101 object
    param = {'block_name': 'BottleNeck',
             'num_block': [3, 4, 23, 3], 'channels': [64, 128, 256, 512],
             'num_classes': 100, 'expansion': 4}
ResNet 152 object
    param = {'block_name': 'BottleNeck',
             'num_block': [3, 4, 36, 3], 'channels': [64, 128, 256, 512],
             'num_classes': 100, 'expansion': 4}       
"""


if __name__ == '__main__':
    """
    
    """
    param = {'input_shape': (256, 512), 'num_classes': 10,
             'encoder_param': {
                 'block_name': 'BasicBlockPreNorm', 'num_block': (3, 4, 5),
                 'channels': (32, 64, 128), 'first_stride': 2,
                 'expansion': 2, 'dropout': 0.
             },
             'decoder_param': {
                 'block_name': 'LinearAtt', 'inner_shape': (2048, 2048),
                 'num_block': 5, 'channels': 512, 'convert_stride': 2,
                 'dropout': 0.
             }
             }
    model = ResNetAtt(**param)
    summary(model, input_size=(1, 3) + param['input_shape'])
    """
    import torch_pruning as tp

    example_inputs = torch.randn((1, 3, 256, 512), device='cuda')
    imp = tp.importance.GroupNormImportance()
    pruned_model = model
    pruner = tp.pruner.MetaPruner(
            pruned_model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            pruning_ratio=0.5,
            ignored_layers=[pruned_model.output],
        )
    pruner.step()
    print(pruned_model)
    summary(pruned_model, input_size=(1, 3, 256, 512))
    """
    # torch.save(model, r'test_model.pth')
