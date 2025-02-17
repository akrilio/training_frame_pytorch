import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .resnet_light import make_nest_block, ConvHead, AttnHead, ResPostNorm


layer_map = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'BN': nn.BatchNorm2d,
    'IN': nn.InstanceNorm2d,
    'LN': nn.LayerNorm,
    'GN': nn.GroupNorm,
    'ConvHead': ConvHead,
    'AttnHead': AttnHead,
    'Rearrange': Rearrange,
    'ResPostNorm': ResPostNorm,
}


def register(func):
    layer_map[func.__name__] = func
    return func


class Conv2dNorm(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight / torch.sum(self.weight, dim=1, keepdim=True)
        return self._conv_forward(x, weight, self.bias)

@register
class ConstGate(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.]))
        self.bias = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        return self.scale * x + self.bias

@register
class SpaceGate(nn.Module):
    def __init__(self, out_channel, layer='Conv', **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(out_channel, 1, kernel_size=1) if 'Conv' in layer
            else nn.Linear(out_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

@register
class DenseGate(nn.Module):
    def __init__(self, out_channel, groups=1, layer='Conv'):
        super().__init__()
        assert out_channel % groups == 0
        self.proj = (nn.Conv2d(out_channel, out_channels=groups, kernel_size=1)
                     if 'Conv' in layer else nn.Linear(out_channel, groups))
        self.gate = nn.Sigmoid()
        if groups == 1 or out_channel:
            self.group_split = nn.Identity()
            self.group_concat = nn.Identity()
        else:
            if 'Conv' in layer:
                self.group_split = Rearrange('b (c g) h w -> (b g) c h w', g=groups)
                self.group_concat = Rearrange('(b g) c h w -> b (c g) h w', g=groups)
            else:
                self.group_split = Rearrange('b n (d g) -> (b g) n d', g=groups)
                self.group_concat = Rearrange('(b g) n d -> b n (d g)', g=groups)

    def forward(self, x):
        a = self.gate(self.proj(x))
        x = self.group_split(x) * self.group_split(a)
        x = self.group_concat(x)
        return x

@register
class GaussianGate(DenseGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = self._forward

    def _forward(self, x, h):
        a = self.gate(- self.proj(x - h) ** 2)
        x = self.group_split(x)
        h = self.group_split(h)
        a = self.group_split(a)
        x = (a - 1) * x + a * h
        x = self.group_concat(x)
        return x

@register
class LayerNormAct(nn.Sequential):
    def __init__(self, block_name, in_channel, out_channel, act=True, **kwargs):
        if block_name == "Conv":
            layers = [
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, **kwargs),
                nn.BatchNorm2d(out_channel),
            ]
        else:
            layers = [
                nn.Linear(in_features=in_channel, out_features=out_channel, **kwargs),
                nn.LayerNorm(out_channel),
            ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)

@register
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channel, out_channel, stride=1, bias=False,
                 gate: str or bool = False, gate_groups=1,
                 norm: str or bool = 'BN', norm_groups=32,
                 act: str or bool = 'ReLU',
                 block_name=None, self_map=None,
                 **kwargs):
        layers = [nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=stride, bias=bias, **kwargs)]
        if gate:
            layers.append(self_map[gate](out_channel, groups=gate_groups, **kwargs))
        if norm == 'BN':
            layers.append(self_map[norm](out_channel))
        elif norm == 'GN':
            layers.append(self_map[norm](norm_groups, out_channel))
        if act:
            layers.append(self_map[act]())
        super().__init__(*layers)


@register
class CTConvBNAct(nn.Sequential):
    def __init__(self, in_channel, out_channel, stride=1, bias=False,
                 gate: str or bool = False, gate_groups=1,
                 norm: str or bool = 'BN', norm_groups=32,
                 act: str or bool = 'ReLU',
                 block_name=None, self_map=None,
                 **kwargs):
        layers = [
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=stride, stride=stride, bias=False),
            # nn.BatchNorm2d(out_channel),  # if me, this is removed, my test shows it is ok, too
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, groups=out_channel, bias=bias, **kwargs),
        ]
        if gate:
            layers.append(self_map[gate](out_channel, groups=gate_groups, **kwargs))
        if norm == 'GN':
            layers.append(self_map[norm](norm_groups, out_channel))
        else:
            layers.append(self_map[norm](out_channel))
        if act:
            layers.append(self_map[act]())
        super().__init__(*layers)

@register
class CTLinearLNAct(nn.Sequential):
    def __init__(self, in_channel, out_channel, num_token, groups=1, bias=False, gate_groups=1,
                 gate: str or bool = False, act: str or bool = 'ReLU', **kwargs):
        super().__init__()
        # channel-wise and token-wise linear layer, channel-wise is dense and token-wise is sparse
        kwargs.pop('block_name', None)
        self_map = kwargs.pop('layer_map', layer_map)
        assert out_channel % groups == 0
        layers = [
            nn.Linear(in_channel, out_channel, bias=bias),
            Rearrange('b n (d g) -> b (g n) d', g=groups),
            nn.Conv1d(groups * num_token, groups * num_token, kernel_size=1, groups=groups),
            Rearrange('b (g n) d -> b n (d g)', g=groups),
        ]
        if gate:
            layers.append(self_map[gate](out_channel, groups=gate_groups, **kwargs))
        layers.append(nn.LayerNorm([num_token, out_channel]))
        if act:
            layers.append(self_map[act]())
        super().__init__(*layers)

@register
class PoolRes(nn.Module):
    def __init__(self, out_channel, kernel_size, padding, stride=1, groups=1, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.add = GaussianGate(out_channel, groups)

    def forward(self, x):
        return self.add(x, self.pool(x))

@register
class Stem(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride, bias=False, **kwargs):
        super().__init__()
        self.in_proj = ConvBNAct(in_channel=in_channel, out_channel=hidden_channel,
                                 kernel_size=3, stride=stride[0], padding=1, **kwargs)
        self.conv = nn.Sequential(
            ConvBNAct(in_channel=hidden_channel, out_channel=hidden_channel // 2,
                      kernel_size=3, stride=1, padding=1, **kwargs),
            ConvBNAct(in_channel=hidden_channel // 2, out_channel=hidden_channel,
                      kernel_size=3, stride=1, padding=1, **kwargs)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.out_proj = ConvBNAct(in_channel=hidden_channel * 2, out_channel=out_channel,
                                  kernel_size=stride[1], stride=stride[1], bias=bias, **kwargs)

    def forward(self, x):
        x = self.in_proj(x)
        x1 = self.conv(x)
        x2 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.out_proj(x)
        return x

@register
class HGBlock(nn.Module):
    def __init__(self, block_name, in_channel, hidden_channel, out_channel, num_layers, stride=1, **kwargs):
        super().__init__()
        channel = in_channel
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = make_nest_block(
                block_name=block_name.copy(),
                in_channel=channel,
                out_channel=hidden_channel,
                **kwargs
            )
            channel = hidden_channel
            self.layers.append(layer)
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        kwargs['gate'] = False
        kwargs['norm'] = 'BN'
        self.out_proj = nn.Sequential(
            ConvBNAct(
                in_channel=in_channel + hidden_channel * num_layers,
                out_channel=out_channel // 2, **kwargs
            ),
            ConvBNAct(
                in_channel=out_channel // 2,
                out_channel=out_channel, **kwargs
            )
        )
        kwargs['kernel_size'] = stride
        kwargs['act'] = False
        if stride == 1 and in_channel == out_channel:
            self.identity = True
            self.in_proj = nn.Identity()
        else:
            self.identity = False
            self.in_proj = ConvBNAct(
                in_channel=in_channel,
                out_channel=in_channel,
                groups=in_channel,
                stride=stride, **kwargs
            )

    def forward(self, x):
        x = self.in_proj(x)
        hidden_fea = [x,]
        for layer in self.layers:
            x = layer(x)
            hidden_fea.append(x)
        x = torch.cat(hidden_fea, dim=1)
        x = self.out_proj(x)
        if self.identity:
            x = hidden_fea[0] + x
        return x

@register
class HGBlockLinear(nn.Module):
    def __init__(self, block_name, in_channel, hidden_channel, out_channel, num_token, num_layers, **kwargs):
        super().__init__()
        self.identity = True if in_channel == out_channel else False
        channel = in_channel
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = make_nest_block(
                block_name=block_name.copy(),
                in_channel=channel,
                out_channel=hidden_channel,
                num_token=num_token,
                **kwargs
            )
            channel = hidden_channel
            self.layers.append(layer)
        self.out_proj = nn.Sequential(
            nn.Linear(in_channel + hidden_channel * num_layers, out_channel // 2),
            nn.LayerNorm([num_token, out_channel // 2]), nn.ReLU(),
            nn.Linear(out_channel // 2, out_channel),
            nn.LayerNorm([num_token, out_channel]), nn.ReLU(),
        )

    def forward(self, x):
        s = x
        hidden_fea = [x, ]
        for layer in self.layers:
            x = layer(x)
            hidden_fea.append(x)
        x = torch.cat(hidden_fea, dim=-1)
        x = self.out_proj(x)
        if self.identity:
            x = s + x
        return x

@register
class HGBlockLiu(nn.Module):
    def __init__(self, block_name, in_channel, hidden_channel, out_channel, num_layers, stride=1, **kwargs):
        super().__init__()
        # use ResPostNorm
        self.squeeze = ConvBNAct(
            in_channel=in_channel,
            out_channel=hidden_channel,
            kernel_size=stride,
            stride=stride,
            act=False
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = make_nest_block(
                block_name=block_name.copy(),
                in_channel=hidden_channel,
                out_channel=hidden_channel,
                **kwargs
            )
            self.layers.append(layer)

        self.extract = ConvBNAct(
            in_channel=hidden_channel * (num_layers + 1),
            out_channel=out_channel,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.squeeze(x)
        hidden_fea = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_fea.append(x)
        x = torch.cat(hidden_fea, dim=1)
        x = self.extract(x)
        return x


class HGNet(nn.Sequential):
    def __init__(self, structure, input_shape=(224, 224), in_channel=3):
        """
        params = {
            'input_shape': input_shape
            'structure': [
                {'block_name': 'Stem', 'num_block': 1, 
                 'hidden_channel': 24, 'out_channel': 128, 'stride': 1},
                {'block_name': ['HGBlock', 'ConvBNAct'], 'num_block': [1, 2, 1], 
                 'hidden_channel': [32, 64, 128], 'out_channel': [128, 256, 512], 'stride': [2, 2, 2]},
                {'block_name': 'ConvHead', 'num_block': 1,
                 'out_channel': num_classes}, # **kwargs
            ]
        }
        """
        self.in_channel = in_channel
        self.input_shape = [*input_shape]
        self.blocks = []
        for params in structure:
            if params['block_name'] in ['Rearrange', 'SeqNet', 'PyramidNet', 'MultiAttnHead']:
                name = params.pop('block_name')
                self.blocks.append(layer_map[name](**params))
            else:
                self._blocks_decode(**params)
        super().__init__(*self.blocks)

    def _blocks_decode(self, block_name, num_block=1, stride: int or list = 1, **kwargs):
        if isinstance(block_name, str):
            block_name = [block_name]

        decode_params = {}
        base_params = {'self_map': layer_map}
        if isinstance(num_block, list):
            for key, value in kwargs.items():
                if isinstance(value, list):
                    decode_params[key] = []
                    for i, n in enumerate(num_block):
                        decode_params[key] += n * [value[i]]
                else:
                    base_params[key] = value
            if isinstance(stride, int):
                stride = len(num_block) * [stride]
            decode_params['stride'] = []
            for i, n in enumerate(num_block):
                decode_params['stride'] += [stride[i]] + (n - 1) * [1]
        else:
            decode_params['stride'] = [stride] + (num_block - 1) * [1]
            num_block = [num_block]
            base_params.update(kwargs)

        for idx in range(sum(num_block)):
            for key, value in decode_params.items():
                base_params[key] = value[idx]
            self.blocks.append(make_nest_block(
                block_name=block_name.copy(),
                in_channel=self.in_channel,
                **base_params
            ))
            self.in_channel = base_params['out_channel']

    def _decode_names(self, block_name, **kwargs):
        if isinstance(block_name[0], str):
            self.blocks.append(make_nest_block(block_name=block_name, **kwargs))
        elif isinstance(block_name[0], list):
            for name in block_name:
                self._decode_names(block_name=name, **kwargs)

@register
class MultiAttnHead(nn.Module):
    def __init__(self, hidden_channel, num_heads: int, **kwargs):
        super().__init__()
        out_channel = kwargs['out_channel']
        kwargs['out_channel'] = hidden_channel
        for key, value in kwargs.items():
            if isinstance(value, list):
                assert len(value) == num_heads
            else:
                kwargs[key] = num_heads * [value]
        head_params = [{k: v[i] for k, v in kwargs.items()} for i in range(num_heads)]
        self.heads = nn.ModuleList()
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        for param in head_params:
            layer = AttnHead(**param)
            self.heads.append(layer)
        self.extract = nn.Linear(hidden_channel * num_heads, out_channel, bias=True)

    def forward(self, features):
        x = []
        for fea, head in zip(features, self.heads):
            fea = self.rearrange(fea)
            x.append(head(fea))
        x = torch.cat(x, dim=-1)
        x = self.extract(x)
        return x


@register
class PyramidNet(HGNet):
    def __init__(self, out_block_index, *args, **kwargs):
        self.index = out_block_index
        super().__init__(*args, **kwargs)

    def forward(self, x):
        hidden_fea = []
        for i, block in enumerate(self):
            x = block(x)
            if i in self.index:
                hidden_fea.append(x)
        return hidden_fea

    """
    arch_configs = {
        'S': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            }
        },
        'M': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            }
        },
        'L': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            }
        },
        'X': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            }
        },
        'H': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            }
        }
    }
    """
