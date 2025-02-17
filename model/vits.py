from torchinfo import summary
from .transformer_unit import *

"""Single scale transformer"""


class BasicViT(nn.Module):
    def __init__(self, image_size, channels, classes, patch_size, depth, dims,
                 num_heads, dim_heads=64, mlp_ratio=4, dropout=0., emb_dropout=0.):
        super().__init__()
        """
        params = {'image_size': (32, 32), 'channels': 3, 'classes': 10,
                  'patch_size': (8, 8), 'depth': 12, 'dims': 192,
                  'num_heads': 3, 'dim_heads': 64, 'mlp_ratio': 4,
                  'dropout': 0., 'emb_dropout': 0.}
        """
        self.name = "BasicViT"
        self.params = {'image_size': image_size, 'channels': channels, 'classes': classes,
                       'patch_size': patch_size, 'depth': depth, 'dims': dims,
                       'num_heads': num_heads, 'dim_heads': dim_heads, 'mlp_ratio': mlp_ratio,
                       'dropout': dropout, 'emb_dropout': emb_dropout}
        i_h, i_w = image_size
        self.p_h, self.p_w = patch_size
        assert i_h % self.p_h == 0 and i_w % self.p_w == 0, 'Image dimensions must be divisible by the patch size.'
        self.n_h, self.n_w = i_h // self.p_h, i_w // self.p_w
        self.classes = classes

        self.dropout = nn.Dropout(emb_dropout)
        self.in_head, self.mlp, self.out_head, d = self.make_head(dims, mlp_ratio, channels, classes, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_h, self.n_w, d))
        self.transformers = self.make_transformers(depth, dims, num_heads, dim_heads, mlp_ratio, dropout)

    def make_head(self, dim, mlp_ratio, channels, classes, dropout):
        d = dim // (self.p_h * self.p_w)
        in_head = nn.Sequential(nn.Conv2d(channels, d, kernel_size=5, padding='same'), nn.BatchNorm2d(d))
        mlp = nn.Sequential(nn.LayerNorm(dim), MLP(dim, dim * mlp_ratio, dim, dropout))
        out_head = nn.Linear(dim, classes)
        return in_head, mlp, out_head, dim

    def make_transformers(self, depth, dims, num_heads, dim_heads, mlp_ratio, dropout):
        transformer_kwargs = {
            "patch_shape": (self.n_h, self.n_w), "dim": dims,
            "num_heads": num_heads, "dim_heads": dim_heads, "mlp_dim": mlp_ratio * dims,
            "attn_blk": EfficientAttention, "mlp_blk": MLP, "dropout": 0.,
        }
        layer = [TransformerBlock(**transformer_kwargs)for _ in range(depth)]
        return nn.Sequential(*layer)

    def to_tokens(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, self.n_h, self.p_h, self.n_w, self.p_w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, self.n_h, self.n_w, self.p_h * self.p_w * c)
        return x

    def forward(self, img):
        x = self.in_head(img)
        x = self.to_tokens(x)
        x = x + self.mlp(x)
        """
        x = self.in_head(img)
        """
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformers(x)
        x = x.mean(dim=[1, 2])
        x = self.out_head(x)
        return x


class PViT(BasicViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PViT"

    def make_transformers(self, depth, dims, num_heads, dim_heads, mlp_ratio, dropout):
        transformer_kwargs = {
            "patch_shape": (self.n_h, self.n_w), "dim": dims,
            "num_heads": num_heads, "dim_heads": dim_heads, "mlp_dim": mlp_ratio * dims,
            "attn_blk": Attention, "mlp_blk": MLP, "dropout": 0.,
        }
        layer = [TransformerBlockParallel(**transformer_kwargs) for _ in range(depth)]
        return nn.Sequential(*layer)


class CViT(BasicViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CViT"

    def make_transformers(self, depth, dims, num_heads, dim_heads, mlp_ratio, dropout):
        transformer_kwargs = {
            "patch_shape": (self.n_h, self.n_w), "dim": dims,
            "num_heads": num_heads, "dim_heads": dim_heads, "mlp_dim": mlp_ratio * dims,
            "attn_blk": CoupleAttention, "mlp_blk": CoupleLinear, "dropout": 0.,
        }
        layer = [TransformerBlock(**transformer_kwargs) for _ in range(depth)]
        return nn.Sequential(*layer)


"""MultiScaled Transformer"""


class UViT(nn.Module):
    def __init__(self, image_size, patch_size, channels, classes, depth, dims, scales,
                 num_heads, dim_heads, mlp_ratio=4, dropout=0., emb_dropout=0., use_rel_pos=False):
        super().__init__()
        """
        params = {'image_size': (32, 32), 'patch_size': (4, 4), 'channels': 3, 'classes': 10,
                  'depth': (3, 6, 12), 'dims': (192, 384, 768), 'scales': (1., 0.5, 0.25),
                  'num_heads': (3, 6, 18), 'dim_heads': (64, 64, 64), 'mlp_ratio': 4, 'mlp_ratio': 4,
                  'dropout': 0., 'emb_dropout': 0., 'use_rel_pos': False}
        """
        self.name = "UViT"
        self.params = {'image_size': image_size, 'patch_size': patch_size, 'channels': channels, 'classes': classes,
                       'depth': depth, 'dims': dims, 'scales': scales,
                       'num_heads': num_heads, 'dim_heads': dim_heads, 'mlp_ratio': mlp_ratio,
                       'dropout': dropout, 'emb_dropout': emb_dropout, 'use_rel_pos': use_rel_pos}
        i_h, i_w = image_size
        self.p_h, self.p_w = patch_size
        assert i_h % self.p_h == 0 and i_w % self.p_w == 0, 'Image dimensions must be divisible by the patch size.'
        self.n_h,  self.n_w = i_h // self.p_h, i_w // self.p_w
        self.classes = classes
        self.use_rel_pos = use_rel_pos

        self.dropout = nn.Dropout(emb_dropout)
        self.in_head, self.mlp, self.out_head, d = self.make_head(dims, mlp_ratio, channels, classes, dropout)
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.n_h, self.n_w, d))

        self.transformer = TransformerBlock
        self.transformer_kwargs = {
            "patch_shape": (self.n_h, self.n_w), "dim": dims[0],
            "num_heads": num_heads, "dim_heads": dim_heads, "mlp_dim": mlp_ratio * dims[0],
            "attn_blk": EfficientAttention, "mlp_blk": MLP, "dropout": dropout, "use_rel_pos": self.use_rel_pos
        }
        self.sampler = ConvSampler
        self.sampler_kwargs = {
            "hw_patch": [i // p for i, p in zip(image_size, patch_size)],
            "in_dim": None, "out_dim": None, "in_scale": 1.0, "out_scale": 0.5
        }
        self.encoder = self.make_transformers(depth, dims, num_heads, dim_heads, mlp_ratio, scales)

    def make_head(self, dims, mlp_ratio, channels, classes, dropout):
        d = dims[0] // (self.p_h * self.p_w)
        in_head = nn.Sequential(nn.Conv2d(channels, d, kernel_size=5, padding='same'), nn.BatchNorm2d(d))
        mlp = nn.Sequential(nn.LayerNorm(dims[0]), MLP(dims[0], dims[0] * mlp_ratio, dims[0], dropout))
        out_head = nn.Linear(dims[-1], classes)
        """
        emb_size = max(self.p_h, self.p_w)
        self.in_head = nn.Sequential(
            nn.Conv2d(channels, dims, kernel_size=emb_size * 2, stride=emb_size, padding=emb_size // 2)
        )
        self.mlp = lambda x: 0
        self.out_head = OutHead(dims[-1], classes)
        """
        return in_head, mlp, out_head, dims[0]

    def make_transformers(self, depth, dims, num_heads, dim_heads, mlp_ratio, scales):
        layers = []
        for i in range(len(depth)):
            self.transformer_kwargs['patch_shape'] = (int(self.n_h * scales[i]), int(self.n_w * scales[i]))
            self.transformer_kwargs['dim'] = dims[i]
            self.transformer_kwargs['num_heads'] = num_heads[i]
            self.transformer_kwargs['dim_heads'] = dim_heads[i]
            self.transformer_kwargs['mlp_dim'] = mlp_ratio * dims[i]
            for _ in range(depth[i]):
                layers.append(self.transformer(**self.transformer_kwargs))
            if i < len(depth) - 1:
                self.sampler_kwargs['in_dim'] = dims[i]
                self.sampler_kwargs['out_dim'] = dims[i + 1]
                self.sampler_kwargs['in_scale'] = scales[i]
                self.sampler_kwargs['out_scale'] = scales[i + 1]
                layers.append(self.sampler(**self.sampler_kwargs))
        return nn.Sequential(*layers)

    def to_tokens(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, self.n_h, self.p_h, self.n_w, self.p_w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, self.n_h, self.n_w, self.p_h * self.p_w * c)
        return x

    def forward(self, img):
        x = self.in_head(img)
        x = self.to_tokens(x)
        x = x + self.mlp(x)
        """
        x = self.in_head(img)
        """
        # x += self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.mean(dim=[1, 2])
        # x = self.out_head(x)
        return x


class ConViT(UViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ConViT"
        self.transformer_kwargs = {
            "patch_shape": (self.n_h, self.n_w), "dim": kwargs['dims'][0],
            "num_heads": kwargs["num_heads"], "dim_heads": kwargs["dim_heads"],
            "mlp_dim": kwargs["mlp_ratio"] * kwargs['dims'][0], "use_rel_pos": self.use_rel_pos,
            "attn_blk": EfficientAttention, "mlp_blk": MLP, "dropout": kwargs["dropout"],
        }


class RViT(UViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RViT"

    def make_head(self, dims, mlp_ratio, channels, classes, dropout):
        d = dims[0] // (self.p_h * self.p_w)
        in_head = InHead(channels, d, num_blk=2, stride=1)
        mlp = nn.Sequential(
            nn.LayerNorm(dims[0]),
            MLP(dims[0], dims[0] * mlp_ratio, dims[0], dropout)
        )
        out_head = nn.Sequential(
            nn.Linear(dims[-1], classes),
            nn.Softmax(dim=1)
        )
        return in_head, mlp, out_head, dims[0]


class BoViT(UViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BoViT"
        self.transformer = BoTransformerBlock

    def make_head(self, dims, mlp_ratio, channels, classes, dropout):
        d = dims[0] // (self.p_h * self.p_w)
        in_head = nn.Sequential(
            nn.Conv2d(channels, d * mlp_ratio, kernel_size=5, padding='same'),
            nn.BatchNorm2d(d * mlp_ratio)
        )
        mlp = nn.Sequential(
            nn.LayerNorm(dims[0] * mlp_ratio),
            MLP(dims[0] * mlp_ratio, dims[0], dims[0] * mlp_ratio, dropout)
        )
        out_head = nn.Sequential(
            nn.Linear(dims[-1] * mlp_ratio, classes),
            nn.Softmax(dim=1)
        )
        return in_head, mlp, out_head, dims[0] * mlp_ratio


class MViT(UViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MViT"
        self.sampler = TransformerSampler


if __name__ == '__main__':
    """
    params = {'image_size': (32, 32), 'channels': 3, 'classes': 10,
              'patch_size': (8, 8), 'depth': 12, 'dims': 384,
              'num_heads': 6, 'dim_heads': 64, 'mlp_ratio': 4,
              'dropout': 0., 'emb_dropout': 0.}
    model = BasicViT(**params)
    
    params = {'image_size': (32, 32), 'patch_size': (1, 1), 'channels': 3, 'classes': 10,
              'depth': (3, 4, 6, 3), 'dims': (32, 96, 192, 384), 'scales': (1., 0.5, 0.25, 0.125),
              'num_heads': 3, 'dim_heads': 32, 'mlp_ratio': 4,
              'dropout': 0., 'emb_dropout': 0., 'use_rel_pos': False}
    model = RViT(**params)
    """
    params = {'image_size': (32, 32), 'patch_size': (1, 1), 'channels': 3, 'classes': 10,
              'depth': (3, 4, 6, 3), 'dims': (64, 128, 256, 512), 'scales': (1., 0.5, 0.25, 0.125),
              'num_heads': (1, 2, 4, 8), 'dim_heads': (64, 64, 64, 64), 'mlp_ratio': 4,
              'dropout': 0., 'emb_dropout': 0., 'use_rel_pos': False}
    model = UViT(**params)
    summary(model, input_size=(1, 3, 32, 32))
