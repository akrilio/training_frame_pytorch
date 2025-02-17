import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math


class LinearSeqBase(nn.Sequential):
    def __init__(self, in_channel, hidden_channel, out_channel, dropout=0., **kwargs):
        layers = [
            nn.Linear(in_channel, hidden_channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channel, out_channel),
            nn.Dropout(dropout),
        ]
        super().__init__(*layers)


class LinearSeqLight(nn.Sequential):
    def __init__(self, in_channel, hidden_channel, out_channel, **kwargs):
        layers = [
            nn.Linear(in_channel, hidden_channel),
            nn.GELU(),
            nn.Linear(hidden_channel, out_channel),
        ]
        super().__init__(*layers)


class CTLinearSeq(nn.Module):
    def __init__(self, num_token, num_head, in_channel, hidden_channel, out_channel, act=True, **kwargs):
        super().__init__()
        hidden_channel = hidden_channel if act else out_channel
        self.activation = nn.GELU() if act else nn.Identity()
        self.shape_0 = (-1, num_head, hidden_channel // num_head, num_token)
        self.shape_1 = (-1, hidden_channel, num_token)
        self.linear_c_0 = nn.Linear(in_channel, hidden_channel)
        self.linear_c_1 = nn.Linear(hidden_channel, out_channel)
        self.weight_t = nn.Parameter(torch.empty((num_head, num_token, num_token)))
        nn.init.kaiming_uniform_(self.weight_t, a=math.sqrt(5))

    def forward(self, x):
        x = self.linear_c_0(x)
        x = self.activation(x)
        x = x.transpose(1, 2).reshape(self.shape_0)
        x = torch.matmul(x, self.weight_t)
        x = x.reshape(self.shape_1).transpose(1, 2)
        x = self.activation(x)
        x = self.linear_c_1(x)
        return x


class CTLinearGate(nn.Module):
    def __init__(self, num_token, num_head, in_channel, hidden_channel, out_channel, act=None, **kwargs):
        super().__init__()
        self.activation = nn.GELU() if act is None else act
        self.gate = nn.Sigmoid()
        self.shape_0 = (-1, num_head, hidden_channel // num_head, num_token)
        self.shape_1 = (-1, hidden_channel, num_token)
        self.linear_c_0 = nn.Linear(in_channel, hidden_channel + 1)
        self.linear_c_1 = nn.Linear(hidden_channel, out_channel + 1)
        self.weight_t = nn.Parameter(torch.empty((num_head, num_token, num_token)))
        nn.init.kaiming_uniform_(self.weight_t, a=math.sqrt(5))

    def forward(self, x):
        x = self.linear_c_0(x)
        a = self.gate(x[..., :1])
        x = self.activation(x[..., 1:])
        x = x.transpose(1, 2).reshape(self.shape_0)
        x = torch.matmul(x, self.weight_t)
        x = x.reshape(self.shape_1).transpose(1, 2)
        x = self.activation(x)
        x = self.linear_c_1(x)
        x[..., 1:] *= self.gate(x[..., :1])
        return x[..., 1:] * a


class CTLinearEX(nn.Module):
    def __init__(self, num_token, in_channel, expansion, out_channel, **kwargs):
        super().__init__()
        self.shape = (-1, num_token, out_channel * expansion)
        self.in_proj = nn.Linear(in_channel, out_channel)
        self.linear_t = nn.Linear(num_token, int(num_token * expansion))
        self.linear_c = nn.Linear(int(out_channel * expansion), out_channel)
        self.activation = nn.GELU()
        # self.out_proj = nn.Linear(num_token, num_token)

    def forward(self, x):
        x = self.in_proj(x)
        x = x.transpose(1, 2)
        x = self.linear_t(x)
        x = self.activation(x)
        x = x.transpose(1, 2).reshape(self.shape)
        x = self.linear_c(x)
        return x


class ConvSeqBase(nn.Sequential):
    def __init__(self, in_channel, hidden_channel, out_channel, stride=1, dropout=0., **kwargs):
        layers = [
            nn.Conv2d(in_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ]
        super().__init__(*layers)


class ConvSeqNeck(nn.Sequential):
    def __init__(self, in_channel, hidden_channel, out_channel, stride=1, dropout=0., **kwargs):
        layers = [
            nn.Conv2d(in_channel, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channel, hidden_channel, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ]
        super().__init__(*layers)


class ConvSeqLight(nn.Sequential):
    def __init__(self, in_channel, hidden_channel, out_channel, stride=1, **kwargs):
        layers = [
            nn.Conv2d(in_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=3, padding=1, bias=False),
        ]
        super().__init__(*layers)


def make_nest_block(block_name: list, self_map: dict, **kwargs):
    cur_name = block_name.pop(0)
    return self_map[cur_name](self_map=self_map, block_name=block_name, **kwargs)


class ChannelGate(nn.Module):
    def __init__(self, block_name, out_channel, **kwargs):
        super().__init__()
        base_block_name = block_name[len(block_name) - 1]
        self.block = make_nest_block(block_name, out_channel=out_channel, **kwargs)
        self.gate = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(out_channel, int(out_channel * 0.25)),
            nn.GELU(),
            nn.Linear(int(out_channel * 0.25), out_channel),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1) if 'Conv' in base_block_name \
            else nn.Conv1d(out_channel, 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.block(x)
        y = self.pool(x)
        shape = y.shape
        y = self.gate(y).reshape(shape)
        return x * y


class SpaceGate(nn.Module):
    def __init__(self, block_name, out_channel, **kwargs):
        super().__init__()
        base_block_name = block_name[len(block_name) - 1]
        self.block = make_nest_block(block_name, out_channel=out_channel + 1, **kwargs)
        self.gate = nn.Sigmoid()
        self.forward = self.conv_forward if 'Conv' in base_block_name else self.linear_forward

    def linear_forward(self, x):
        x = self.block(x)
        return x[..., 1:] * self.gate(x[..., :1])

    def conv_forward(self, x):
        x = self.block(x)
        return x[:, 1:] * self.gate(x[:, :1])


class PreSpaceGate(nn.Module):
    def __init__(self, block_name, in_channel, **kwargs):
        super().__init__()
        base_block_name = block_name[len(block_name) - 1]
        self.block = make_nest_block(block_name, in_channel=in_channel, **kwargs)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=3, padding=1) if 'Conv' in base_block_name
            else nn.Linear(in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x) * self.gate(x)
        return x


class DoubleSpaceGate(nn.Module):
    def __init__(self, block_name, in_channel, out_channel, **kwargs):
        super().__init__()
        base_block_name = block_name[len(block_name) - 1]
        self.block = make_nest_block(block_name, in_channel=in_channel, out_channel=out_channel, **kwargs)
        if 'Conv' in base_block_name:
            self.in_gate = nn.Sequential(
                nn.Conv2d(in_channel, 1, kernel_size=3, padding=1), nn.Sigmoid())
            self.out_gate = nn.Sequential(
                nn.Conv2d(out_channel, 1, kernel_size=3, padding=1), nn.Sigmoid())
        else:
            self.in_gate = nn.Sequential(nn.Linear(in_channel, 1), nn.Sigmoid())
            self.out_gate = nn.Sequential(nn.Linear(out_channel, 1), nn.Sigmoid())

    def forward(self, x):
        a = self.in_gate(x)
        x = self.block(x)
        a = self.out_gate(x) * a
        return x * a


class SpaceGateRes(nn.Module):
    def __init__(self, block_name, in_channel, out_channel, **kwargs):
        super().__init__()
        base_block_name = block_name[len(block_name) - 1]
        self.block = make_nest_block(block_name, in_channel=in_channel, out_channel=out_channel, **kwargs)
        self.block.add = self.add
        self.forward = self.block.forward
        self.gate = nn.Sigmoid()
        if 'Conv' in base_block_name:
            self.in_proj = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1, bias=False)
            self.out_proj = nn.Conv2d(out_channel, 1, kernel_size=3, padding=1)
        else:
            self.in_proj = nn.Linear(in_channel, 1)  # , bias=False
            self.out_proj = nn.Linear(out_channel, 1)

    def add(self, x, h):
        a = self.gate(self.in_proj(x)) * self.gate(self.out_proj(h))
        return x + a * h


class LinearResPreNorm(nn.Module):
    def __init__(self, block_name, in_channel, out_channel, **kwargs):
        super().__init__()
        assert in_channel == out_channel
        self.block = make_nest_block(block_name, in_channel=in_channel, out_channel=out_channel, **kwargs)
        self.norm = nn.LayerNorm(in_channel)
        self.add = torch.add

    def forward(self, x):
        return self.add(x, self.block(self.norm(x)))


class ResFrame(nn.Module):
    def __init__(self, block_name, in_channel, out_channel, stride=1, **kwargs):
        super().__init__()
        base_block_name = block_name[len(block_name) - 1]
        self.block = make_nest_block(
            block_name=block_name, in_channel=in_channel, out_channel=out_channel, stride=stride, **kwargs
        )
        if 'Conv' in base_block_name:
            self.norm = nn.BatchNorm2d
            self.layer = nn.Conv2d
        elif 'Linear' in base_block_name:
            self.norm = nn.LayerNorm
            self.layer = nn.Linear

        if stride == 1 and in_channel == out_channel:
            self.shortcut = nn.Identity()
        else:
            if 'Light' in base_block_name:
                self.shortcut = self.layer(in_channel, out_channel, kernel_size=stride, stride=stride, bias=False)
            else:
                self.shortcut = nn.Sequential(
                    self.layer(in_channel, out_channel, kernel_size=stride, stride=stride, bias=False),
                    self.norm(out_channel)
                )


class ResAct(ResFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.shortcut(x) + self.block(x))


class ResPreNorm(ResFrame):
    def __init__(self, in_channel, **kwargs):
        super().__init__(in_channel=in_channel, **kwargs)
        self.norm = self.norm(in_channel)

    def forward(self, x):
        return self.shortcut(x) + self.block(self.norm(x))


class ResPostNorm(ResFrame):
    def __init__(self, out_channel, **kwargs):
        super().__init__(out_channel=out_channel, **kwargs)
        self.norm = self.norm(out_channel)

    def forward(self, x):
        return self.norm(self.shortcut(x) + self.block(x))


class CTLinear(nn.Module):
    def __init__(self, block_name: list, num_token, in_channel, out_channel, inner_shape, dropout=0.):
        super().__init__()
        n_in, e_in = inner_shape
        curr_name = block_name.pop(0)
        block = eval(curr_name)
        params = {'in_channel': in_channel, 'hidden_channel': e_in, 'out_channel': out_channel, 'dropout': dropout}
        self.block_c = block(block_name=block_name.copy(), **params)
        params = {'in_channel': num_token, 'hidden_channel': n_in, 'out_channel': num_token, 'dropout': dropout}
        self.block_t = block(block_name=block_name.copy(), **params)

    def forward(self, x):
        x = self.block_c(x)
        x = x.transpose(1, 2)
        x = self.block_t(x)
        x = x.transpose(1, 2)
        return x


class TransLinear(nn.Module):
    def __init__(self, block_name: list, **kwargs):
        super().__init__()
        block_c = block_name.copy()
        block_c.append('LinearSeqLight')
        self.block_c = make_nest_block(block_c, **kwargs)
        block_t = block_name.copy()
        block_t.append('CTLinearSeq')
        self.block_t = make_nest_block(block_t, act=False, **kwargs)

    def forward(self, x):
        x = self.block_c(x)
        x = self.block_t(x)
        return x

"""
class AttnHead(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, **kwargs):
        super().__init__()
        self.num_class = out_channel
        self.attn = nn.Sequential(nn.Linear(in_channel, out_channel), nn.Softmax(dim=1))
        self.proj = nn.Linear(in_channel, 1, bias=bias)

    def forward(self, x):
        a = self.attn(x)
        x = a.transpose(1, 2) @ self.proj(x)
        # x = self.proj(x)
        return x.view(-1, self.num_class)
"""

class AttnHead(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel: int = None,
                 head=1, group=1, bias=True, **kwargs):
        super().__init__()
        hidden_channel = hidden_channel if hidden_channel is not None else out_channel
        assert hidden_channel % (head * group) == 0
        self.head = head
        self.attn = nn.Sequential(
            nn.Linear(in_channel, head * group),
            nn.Softmax(dim=1),
            Rearrange('b n (g h) -> b g h n', g=group)
        )
        self.proj = nn.Sequential(
            nn.Linear(in_channel, hidden_channel // head, bias=bias),
            Rearrange('b n (g d) -> b g n d', g=group)
        )
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, out_channel, bias=bias)
        )
        
    def forward(self, x):
        a = self.attn(x)
        x = self.proj(x)
        x = self.fc(a @ x)
        return x


class LinearAttn(nn.Module):
    def __init__(self, block_name, out_channel, att_head=None, **kwargs):
        super().__init__()
        hidden_channel = att_head if att_head is not None else int(out_channel * 0.25)
        self.block = make_nest_block(block_name, out_channel=out_channel, **kwargs)
        self.attn = nn.Sequential(
            nn.Linear(out_channel, hidden_channel),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block(x)
        a = self.attn(x)
        return a @ (a.transpose(1, 2) @ x)


class ConvAttn(LinearAttn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = self.block(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        a = self.attn(x)
        x = a @ (a.transpose(1, 2) @ x)
        return x.transpose(1, 2).reshape(b, c, h, w)


class DenseHead(nn.Sequential):
    layers = []

    def __init__(self, in_channel, out_channel, **kwargs):
        self.layers.extend([nn.Flatten(start_dim=1), nn.Linear(in_channel, out_channel)])
        super().__init__(*self.layers)


class LinearHead(DenseHead):
    def __init__(self, in_channel, out_channel, **kwargs):
        self.layers.extend([Rearrange('b n c -> b c n'), nn.AdaptiveAvgPool1d(1)])
        super().__init__(in_channel, out_channel, **kwargs)


class ConvHead(DenseHead):
    def __init__(self, in_channel, out_channel, **kwargs):
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        super().__init__(in_channel, out_channel, **kwargs)


layer_map = {name: layer for name, layer in globals().items()}


class ResNetAtt(nn.Module):
    def __init__(self, input_shape, encoder_param=None, decoder_param=None, head='DenseHead', num_classes=10):
        super().__init__()
        self.name = "ResNetAtt"
        self.params = {'input_shape': input_shape, 'num_classes': num_classes,
                       'encoder_param': encoder_param, 'decoder_param': decoder_param,
                       'head': head}
        self.input_shape = [input_shape[0], input_shape[1]]
        self.in_channel = 3
        self.blocks = []
        if isinstance(encoder_param, dict):
            self._make_encoder(**encoder_param)
        if isinstance(decoder_param, dict):
            self._make_decoder(**decoder_param)
        elif isinstance(head, str) and 'Attn' in head:
            self.blocks.append(Rearrange('b c h w -> b (h w) c'))
        if head == 'DenseHead':
            self.in_channel *= self.input_shape[0] * self.input_shape[1]
        self.blocks = nn.Sequential(*self.blocks)
        self.head = eval(head)(self.in_channel, num_classes)

    def _make_encoder(self, block_name, num_block, channels, expansion, first_stride, dropout=0.):
        strides = []
        out_channels = []
        for num, channel in zip(num_block, channels):
            strides += [2] + [1] * (num - 1)
            out_channels += [channel] * num
        self.blocks.append(nn.Sequential(
            nn.Conv2d(self.in_channel, out_channels[0], kernel_size=first_stride, stride=first_stride),
            nn.BatchNorm2d(out_channels[0])
        ))
        strides[0] = 1
        self.in_channel = out_channels[0]
        self.input_shape = [i // first_stride for i in self.input_shape]
        for stride, out_channel in zip(strides, out_channels):
            params = {
                'layer_map': layer_map,
                'in_channel': self.in_channel,
                'hidden_channel': int(out_channel * expansion),
                'out_channel': out_channel,
                'stride': stride, 'dropout': dropout
            }
            self.blocks.append(make_nest_block(block_name.copy(), **params))
            self.in_channel = out_channel
        self.input_shape = [i // 2 ** (len(num_block) - 1) for i in self.input_shape]

    def _make_decoder(self, block_name, num_block, channel, convert_stride, **kwargs):
        self.blocks.append(nn.Sequential(
            nn.Conv2d(self.in_channel, channel, kernel_size=convert_stride, stride=convert_stride),
            nn.BatchNorm2d(channel),
            Rearrange('b c h w -> b (h w) c')
        ))
        self.in_channel = channel
        self.input_shape = [i // convert_stride for i in self.input_shape]
        h, w = self.input_shape
        params = {'layer_map': layer_map, 'num_token': h * w, 'in_channel': channel, 'out_channel': channel}
        for _ in range(num_block):
            self.blocks.append(make_nest_block(block_name.copy(), **params, **kwargs))

    def forward(self, x):
        x = self.blocks(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    """
            'decoder_param': {
                 'block_name': ['CTLinear', 'ResPreNorm', 'SpaceGate', 'LinearSeqLight'],
                 'num_block': 4, 'convert_stride': 2, 'channel': 512,
                 'inner_shape': (2048, 2048), 'dropout': 0.,
            }
            'decoder_param': {
                 'block_name': ['ResPreNorm', 'SpaceGate', 'CTLinearSeq'],
                 'num_block': 4, 'convert_stride': 2, 'channel': 512,
                 'hidden_channel': 2048, 'num_head': 4, 'dropout': 0.,
            },
            'encoder_param': {
                 'block_name': ['ResPreNorm', 'ConvSeqLight'],
                 'num_block': (2, 2, 3), 'channels': (32, 64, 128),
                 'first_stride': 2, 'expansion': 2, 'dropout': 0.
             },
            'decoder_param': {
                 'block_name': ['SpaceGateRes', 'LinearResPreNorm', 'CTLinearSeq'],
                 'num_block': 8, 'convert_stride': 2, 'channel': 512,
                 'hidden_channel': 2048, 'num_head': 4, 'dropout': 0.,
             },
    """
    param = {'input_shape': (256, 512), 'num_classes': 10,
             'encoder_param': {
                 'block_name': ['ResAct', 'ConvSeqNeck'],
                 'num_block': (3, 4, 6, 4), 'channels': (128, 256, 512, 1024),
                 'first_stride': 2, 'expansion': 0.25, 'dropout': 0.
             },
             'head': 'ConvHead',
             }
    # torch.autograd.set_detect_anomaly(True)
    model = ResNetAtt(**param)
    summary(model, input_size=(1, 3, 256, 512))
    # torch.save(model, r'test_model.pth')
