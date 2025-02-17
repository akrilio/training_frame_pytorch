import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from .resnet import BasicBlock, BottleNeck
from .kan import MyKANLinear2, Gaussian


class InvLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input + self.bias, self.weight)


class CoupleLinear(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, activation=nn.GELU(), bias=True):
        super().__init__()
        assert dim == out_dim
        self.in_weight = nn.Parameter(torch.randn((hidden_dim, dim)) / math.sqrt(hidden_dim))
        self.out_weight = nn.Parameter(torch.randn((dim, hidden_dim)) / math.sqrt(hidden_dim))
        self.alpha = nn.Parameter(torch.randn(1))
        self.activation = activation
        if bias:
            self.in_bias = nn.Parameter(torch.randn(hidden_dim))
            self.out_bias = nn.Parameter(torch.randn(dim))
        else:
            self.register_parameter('in_bias', None)
            self.register_parameter('out_bias', None)

    def in_proj(self, x):
        return F.linear(x, self.in_weight, self.in_bias)

    def out_proj(self, x):
        return F.linear(x,  self.alpha * self.in_weight.T + self.out_weight, self.out_bias)

    def forward(self, x):
        x = F.linear(x, self.in_weight, self.in_bias)
        x = self.activation(x)
        x = F.linear(x,  self.alpha * self.in_weight.T + self.out_weight, self.out_bias)
        return x


class MLP(nn.Sequential):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0., activation=nn.GELU()):
        layers = [
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        ]
        super().__init__(*layers)


class DWConvMLP(nn.Sequential):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        layers = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        ]
        self.mlp = nn.Sequential(*layers)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding='same', groups=dim)

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.dw_conv(x)
        x = x.transpose(1, 3)
        x = self.mlp(x)
        return x


class KAN(nn.Sequential):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        layers = [
            MyKANLinear2(dim, out_dim, grid_size=5, spline_order=0),
            nn.Dropout(dropout)
        ]
        super().__init__(*layers)


class AttentionFrame(nn.Module):
    def __init__(self, patch_shape: tuple, num_heads: int, dim_heads: int,
                 dropout: float, use_rel_pos: bool = False) -> None:
        super().__init__()
        self.h, self.w = patch_shape
        self.internal_dim = dim_heads * num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * patch_shape[0] - 1, dim_heads))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * patch_shape[1] - 1, dim_heads))

    def cos_sin_pos_emb(self, num_heads, dim_heads):
        assert dim_heads % 4 == 0
        k = dim_heads // 4
        omega_t = torch.pow(1 / 10000, torch.arange(k, device='cuda') * 4 / dim_heads).reshape(1, -1)
        xx, yy = torch.meshgrid(torch.arange(self.h, device='cuda'), torch.arange(self.w, device='cuda'))
        xx = xx.reshape(1, 1, -1, 1)
        yy = yy.reshape(1, 1, -1, 1)
        cos_x = torch.cos(xx * omega_t)
        sin_x = torch.sin(xx * omega_t)
        cos_y = torch.cos(yy * omega_t)
        sin_y = torch.sin(yy * omega_t)
        pos_emb = torch.cat([cos_x, sin_x, cos_y, sin_y], dim=-1).repeat(1, num_heads, 1, 1)
        return pos_emb / math.sqrt(dim_heads / 2)

    """
    def cos_sin_pos_emb(self, num_heads, dim_heads):
        assert dim_heads % 2 == 0
        k = dim_heads // 2
        omega_t = torch.pow(1 / 10000, torch.arange(k, device='cuda') * 2 / dim_heads).reshape(1, -1)
        # torch.tensor(math.pi / (2 * self.h * self.w), device='cuda').reshape(1, -1)
        l = torch.arange(self.h * self.w, device='cuda').reshape(-1, 1)
        cos_l = torch.cos(l * omega_t)
        sin_l = torch.sin(l * omega_t)
        pos_emb = torch.cat([cos_l, sin_l], dim=-1)
        print(pos_emb.max(), pos_emb.min())
        return pos_emb / math.sqrt(dim_heads / 2)
    """
    def get_rel_pos(self, size, rel_pos):
        q_coords = torch.arange(size)[:, None]
        k_coords = torch.arange(size)[None, :]
        relative_coords = (q_coords - k_coords) + (size - 1)
        return rel_pos[relative_coords.long()]

    def add_rel_pos(self, attn, q):
        Rh = self.get_rel_pos(self.h, self.rel_pos_h)
        Rw = self.get_rel_pos(self.w, self.rel_pos_w)
        b, n, _, dim = q.shape
        r_q = q.reshape(b * n, self.h, self.w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        attn = ((attn.view(b * n, self.h, self.w, self.h, self.w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :])
                .view(b, n, self.h * self.w, self.h * self.w))
        return attn

    def _separate_heads(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, self.h * self.w, self.num_heads, self.internal_dim // self.num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        return x.reshape(-1, self.h, self.w, self.internal_dim)  # B x N_tokens x C


class Attention(AttentionFrame):
    def __init__(self, patch_shape: tuple, embedding_dim: int, num_heads: int, dim_heads: int,
                 dropout: float, use_rel_pos: bool = False, bias=True) -> None:
        super().__init__(patch_shape, num_heads, dim_heads, dropout, use_rel_pos)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.abs_pos = nn.Parameter(self.cos_sin_pos_emb(num_heads, dim_heads))
        self.scale = torch.tensor(1 / math.sqrt(dim_heads))

    def forward(self, x: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(x)
        q = self.q_proj(x) + v
        k = self.k_proj(x) + v

        # Separate into heads
        q = self._separate_heads(q)
        k = self._separate_heads(k) + self.abs_pos
        v = self._separate_heads(v) + self.abs_pos

        # Attention
        attn = q @ k.transpose(2, 3) * self.scale  # B x N_heads x N_tokens x N_tokens
        if self.use_rel_pos:
            attn = self.add_rel_pos(attn, q)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class EfficientAttention(AttentionFrame):
    def __init__(self, patch_shape: tuple, embedding_dim: int, num_heads: int, dim_heads: int,
                 dropout: float, use_rel_pos: bool = False, bias=True) -> None:
        super().__init__(patch_shape, num_heads, dim_heads, dropout, use_rel_pos)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        pos_emb = torch.rand(1, 1, self.h * self.w, dim_heads).repeat(1, num_heads, 1, 1) / 10
        # self.cos_sin_pos_emb(num_heads, dim_heads)
        self.q_pos = nn.Parameter(pos_emb)
        self.k_pos = nn.Parameter(pos_emb.transpose(2, 3))
        self.q_scale = nn.Parameter(torch.tensor(1 / (2 * dim_heads) ** 0.5))  # 0.2
        self.k_scale = nn.Parameter(torch.tensor(1 / (2 * self.h * self.w) ** 0.5))  # 0.2

    def forward(self, x: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(x)
        q = self.q_proj(x) + v
        k = self.k_proj(x) + v

        # Separate into heads
        v = self._separate_heads(v)
        q = self._separate_heads(q)
        k = self._separate_heads(k).transpose(2, 3)

        q = torch.softmax(q * self.q_scale, dim=-1)
        k = torch.softmax(k * self.k_scale, dim=-1)
        q_pos = self.q_pos / torch.sum(self.q_pos, dim=-1, keepdim=True)
        k_pos = self.k_pos / torch.sum(self.k_pos, dim=-1, keepdim=True)
        out = q @ (k @ v) + q_pos @ (k_pos @ v)
        # / torch.sum(self.q_pos @ self.k_pos, dim=-1, keepdim=True)

        # Get output
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class ExternalAttention(AttentionFrame):
    def __init__(self, patch_shape: tuple, embedding_dim: int, num_heads: int, dim_heads: int,
                 dropout: float, use_rel_pos: bool = False, bias=True) -> None:
        super().__init__(patch_shape, num_heads, dim_heads, dropout, use_rel_pos)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        pos_emb = torch.rand(1, 1, self.h * self.w, dim_heads).repeat(1, num_heads, 1, 1) / 10
        # self.pos = self.cos_sin_pos_emb(num_heads, dim_heads)
        self.q_pos = nn.Parameter(pos_emb)
        self.k_pos = nn.Parameter(pos_emb.transpose(2, 3))

    def forward(self, x: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(x)

        # Separate into heads
        v = self._separate_heads(v)
        q = self.q_pos / torch.sum(self.q_pos, dim=-1, keepdim=True)
        k = self.k_pos / torch.sum(self.k_pos, dim=-1, keepdim=True)
        out = q @ (k @ v)  # / (q @ torch.sum(k, dim=-1, keepdim=True))

        # Get output
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class FreeAttention(AttentionFrame):
    def __init__(self, patch_shape: tuple, embedding_dim: int, num_heads: int, dim_heads: int,
                 dropout: float, use_rel_pos: bool = False, bias=True) -> None:
        super().__init__(patch_shape, num_heads, dim_heads, dropout, use_rel_pos)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.bias = nn.Parameter(torch.Tensor(1, num_heads, self.h * self.w, self.h * self.w))
        nn.init.xavier_uniform_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(x)
        q = self.q_proj(x)
        k = self.k_proj(x)

        # Separate into heads
        v = self._separate_heads(v)
        q = self._separate_heads(q)
        k = self._separate_heads(k)

        q = torch.sigmoid(q)
        k = torch.exp(k)
        b = torch.exp(self.bias)
        out = q * (b @ (k * v) / (b @ k))

        # Get output
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class LinearAttention(AttentionFrame):
    def __init__(self, patch_shape: tuple, embedding_dim: int, num_heads: int, dim_heads: int,
                 dropout: float, use_rel_pos: bool = False, bias=True) -> None:
        super().__init__(patch_shape, num_heads, dim_heads, dropout, use_rel_pos)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.scale = torch.tensor(1 / (4 * self.h * self.w * dim_heads) ** 0.5)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(x)
        q = self.q_proj(x) + v
        k = self.k_proj(x) + v

        # Separate into heads
        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        out = q @ (k.transpose(2, 3) @ v) * self.scale
        out = self.activation(out)

        # Get output
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class CoupleAttention(AttentionFrame):
    def __init__(self, patch_shape: tuple, embedding_dim: int, num_heads: int, dim_heads: int,
                 dropout: float, use_rel_pos: bool = False, bias=True) -> None:
        super().__init__(patch_shape, num_heads, dim_heads, dropout, use_rel_pos)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim, bias=bias)
        self.couple = CoupleLinear(embedding_dim, self.internal_dim, nn.Identity())

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        v = self.couple.in_proj(v)
        q = self.q_proj(q) + v
        k = self.k_proj(k) + v

        # Separate into heads
        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        if self.use_rel_pos:
            attn = self.add_rel_pos(attn, q)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.couple.out_proj(out)
        return out


class CoupleAttentionSampler(AttentionFrame):
    def __init__(self, patch_shape: tuple, in_dim: int, down_dim: int, up_dim: int, in_tokens: int, out_tokens: int,
                 num_heads: int, dim_heads: int, dropout: float, bias=True) -> None:
        super().__init__(patch_shape, num_heads, dim_heads, dropout)

        self.sampler = nn.Linear(in_tokens, out_tokens)
        assert up_dim % num_heads == 0, "num_heads must divide up_dim."

        self.q_proj = nn.Linear(in_dim, self.internal_dim, bias=bias)
        self.k_proj = nn.Linear(in_dim, self.internal_dim, bias=bias)
        self.v_proj = nn.Linear(in_dim, self.internal_dim)

        self.v_down = nn.Linear(self.internal_dim, down_dim)
        self.v_up = nn.Linear(down_dim, up_dim)
        self.attn = None

    def down_sample(self, x: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Separate into heads
        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        # Attention
        q = self.sampler(q.permute(0, 1, 3, 2))  # B x N_heads x C_per_head x out_tokens
        attn = k @ q
        attn = attn.permute(0, 1, 3, 2)  # B x N_heads x out_tokens x in_tokens
        attn = attn / math.sqrt(k.shape[-1])
        self.attn = torch.softmax(attn, dim=-1)

        # Get output
        out = self.attn @ v
        out = self._recombine_heads(out)
        out = self.v_down(out)
        return out

    def up_sample(self, v: Tensor) -> Tensor:
        v = self._separate_heads(v)
        out = self.attn.permute(0, 1, 3, 2) @ v
        out = self._recombine_heads(out)
        up = self.v_up(out)
        return up


"""
class TransformerBlock(nn.Module):
    def __init__(self, patch_shape, dim, num_heads, dim_heads, mlp_dim, dropout=0., use_rel_pos=False):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = EfficientAttention(patch_shape, dim, num_heads, dim_heads, dropout, use_rel_pos)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dim, dropout)

    def forward(self, x):
        y = self.attn_norm(x)
        x = self.attn(y) + x
        y = self.mlp_norm(x)
        x = self.mlp(y) + x
        return x
"""


class TransformerFrame(nn.Module):
    def __init__(self, patch_shape, dim, num_heads, dim_heads, mlp_dim,
                 attn_blk, mlp_blk, dropout=0., use_rel_pos=False):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = attn_blk(patch_shape, dim, num_heads, dim_heads, dropout, use_rel_pos)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = mlp_blk(dim, mlp_dim, dim, dropout,)  # Gaussian(h=0.1) nn.GELU()


class TransformerBlock(TransformerFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Couple: attn_blk=CoupleAttention, mlp_blk=CoupleLinear
        # Kan: attn_blk=Attention, mlp_blk=KAN
        # DW: attn_blk=Attention, mlp_blk=DWConvMLP

    def forward(self, x):
        y = self.attn_norm(x)
        x = self.attn(y) + x
        y = self.mlp_norm(x)
        x = self.mlp(y) + x
        return x


class TransformerBlockPost(TransformerFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = self.attn(x, x, x) + x
        x = self.attn_norm(x)
        x = self.mlp(x) + x
        x = self.mlp_norm(x)
        return x


class TransformerBlockParallel(TransformerFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = self.attn_norm(x)
        x = self.attn(x, x, x) + self.mlp(x) + x
        return x


class BoTransformerBlock(nn.Module):
    def __init__(self, patch_shape, dim, num_heads, dim_heads, mlp_dim, dropout=0., use_rel_pos=False):
        super().__init__()
        self.dim = dim
        self.bottle_attn = nn.Sequential(
            nn.Linear(mlp_dim, dim),
            nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            EfficientAttention(patch_shape, dim, num_heads, dim_heads, dropout, use_rel_pos),
            nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, mlp_dim),
            nn.LayerNorm(mlp_dim), nn.Dropout(dropout),
        )
        self.out = nn.GELU()

    def forward(self, x):
        return self.out(x + self.bottle_attn(x))


class InHead(nn.Module):
    def __init__(self, in_channel, dim, num_blk, stride):
        super().__init__()
        if num_blk > 0:
            self.encoder = self.make_bottleneck(in_channel, dim, num_blk, stride)
        else:
            self.encoder = nn.Sequential(nn.Conv2d(in_channel, dim, kernel_size=5, padding='same'), nn.BatchNorm2d(dim))

    def make_bottleneck(self, in_channel, dim, num_blk, stride):
        layers = []
        for _ in range(num_blk - 1):
            layers.append(BottleNeck(in_channel, dim, 1))
            in_channel = dim * BottleNeck.expansion
        layers.append(BasicBlock(in_channel, dim, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class OutHead(nn.Module):
    """Seq pool"""
    def __init__(self, dim, classes):
        super().__init__()
        self.a_proj = nn.Sequential(nn.Linear(dim, 1), nn.Softmax(dim=1))
        self.predict = nn.Sequential(nn.Linear(dim, classes), nn.Softmax(dim=1))

    def forward(self, x):
        b, _, _, d = x.shape
        x = x.reshape(b, -1, d)
        attn = self.a_proj(x)
        x = (x.transpose(1, 2) @ attn).squeeze(-1)
        x = self.predict(x)
        return x


class ConvSampler(nn.Module):
    def __init__(self, hw_patch, in_dim, out_dim, in_scale=1.0, out_scale=0.5):
        super().__init__()
        self.n_h_in,  self.n_w_in = [int(i * in_scale) for i in hw_patch]
        self.n_h_out, self.n_w_out = [int(i * out_scale) for i in hw_patch]
        self.in_ch, self.out_ch = in_dim, out_dim
        rel_scale = out_scale / in_scale
        if rel_scale <= 1:
            self.sampler = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=int(1 / rel_scale), stride=int(1 / rel_scale))
        else:
            self.sampler = nn.ConvTranspose2d(self.in_ch, self.out_ch, int(rel_scale), stride=int(rel_scale))
    """
    def forward(self, x):
        x = x.reshape(-1, self.n_h_in, self.n_w_in, self.in_ch).permute(0, 3, 1, 2)
        x = self.sampler(x)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.n_h_out, self.n_w_out, self.out_ch)
        return x
    """

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.sampler(x)
        return x.transpose(1, 3)


class AttentionSampler(AttentionFrame):
    def __init__(self, hw_patch: tuple, in_dim: int, out_dim: int, num_heads: int, dim_heads: int,
                 dropout: float, in_scale=1.0, out_scale=0.5, bias=True, use_rel_pos=False) -> None:
        patch_shape = (int(hw_patch[0] * out_scale), int(hw_patch[1] * out_scale))
        super().__init__(patch_shape, num_heads, dim_heads, dropout, use_rel_pos)
        self.q_proj = nn.Sequential(
            ConvSampler(hw_patch, in_dim, out_dim, in_scale, out_scale),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, self.internal_dim, bias=bias)
        )
        self.k_proj = nn.Sequential(
            ConvSampler(hw_patch, in_dim, out_dim, in_scale, out_scale),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, self.internal_dim, bias=bias)
        )
        self.v_proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, self.internal_dim, bias=bias)
        )

        self.x_proj = ConvSampler(hw_patch, in_dim, out_dim, in_scale, out_scale)
        self.out_proj = nn.Linear(self.internal_dim, out_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        x = self.x_proj(v)
        v = self.v_proj(x)
        q = self.q_proj(q) + v
        k = self.k_proj(k) + v
        # Separate into heads
        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out + x


class TransformerSampler(nn.Module):
    def __init__(self, hw_patch, in_dim, out_dim, num_heads, dim_heads, mlp_dim,
                 in_scale, out_scale, dropout=0., use_rel_pos=False):
        super().__init__()
        self.attn = AttentionSampler(hw_patch, in_dim, out_dim, num_heads, dim_heads,
                                     dropout, in_scale, out_scale, use_rel_pos)
        self.mlp_norm = nn.LayerNorm(out_dim)
        self.mlp = MLP(out_dim, mlp_dim, out_dim, dropout)
        # CoupleLinear(out_dim, mlp_dim, out_dim, dropout)

    def forward(self, x):
        x = self.attn(x, x, x)
        y = self.mlp_norm(x)
        x = self.mlp(y) + x
        return x


if __name__ == '__main__':
    params = {'patch_shape': (16, 16), 'embedding_dim': 384,
              'num_heads': 6, 'dim_heads': 64, 'dropout': 0., 'use_rel_pos': False}
    layer = EfficientAttention(**params)
    print(layer.abs_pos.shape)
    print(layer.abs_pos.max(), layer.abs_pos.min())
    print(layer.abs_pos @ layer.abs_pos.transpose(-1, -2))
