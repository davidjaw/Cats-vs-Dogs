import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torchvision
from torchvision.models import ResNet50_Weights


class Image2Seq(nn.Module):
    def __init__(self, img_size, N, channels=3, emb_size=128, batch_first=True):
        super(Image2Seq, self).__init__()
        patch_size = img_size // N
        assert patch_size * N == img_size, "img_size must be divisible by N"

        self.patch_embedding = nn.Conv2d(channels, emb_size, kernel_size=patch_size, stride=patch_size, groups=4)
        self.pos_embedding = nn.Parameter(torch.randn(1, N * N, emb_size))
        self.batch_first = batch_first

    def forward(self, x):
        # input: [B, C, H, W]
        x = self.patch_embedding(x)       # [B, emb_size, N, N]
        x = x.flatten(2).transpose(1, 2)  # [B, N * N, emb_size]
        x = x + self.pos_embedding        # [B, N * N, emb_size]
        if not self.batch_first:
            # [N * N, B, emb_size]
            x = x.transpose(0, 1)
        return x


class ViTEncoderLayer(nn.Module):
    def __init__(
            self,
            img_size,
            N,
            in_channels,
            emb_size,
            nhead,
            dropout=.1,
            batch_first=True
    ):
        super(ViTEncoderLayer, self).__init__()
        self.img2seq = Image2Seq(img_size, N, in_channels, emb_size, batch_first)
        transformer_encoder_func = partial(nn.TransformerEncoderLayer, d_model=emb_size, nhead=nhead,
                                           dim_feedforward=emb_size, dropout=dropout, batch_first=batch_first)
        self.transformer_encoder = nn.Sequential(
            transformer_encoder_func(),
            transformer_encoder_func(),
            transformer_encoder_func(),
        )

    def forward(self, x):
        x = self.img2seq(x)
        x = self.transformer_encoder(x)
        return x


class ResNetBlk(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            act=nn.ReLU,
            batch_norm=True,
            down_sample=False,
            group=False
    ):
        super(ResNetBlk, self).__init__()
        g = 4 if group else 1

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_c) if batch_norm else nn.Identity(),
            act(),
            nn.Conv2d(in_c, out_c // 4, kernel_size=1, stride=1, bias=False, groups=g),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_c // 4) if batch_norm else nn.Identity(),
            act(),
            nn.Conv2d(out_c // 4, out_c // 4, kernel_size=3, stride=2 if down_sample else 1, padding=1, bias=False, groups=g),
        )
        self.conv3 = nn.Conv2d(out_c // 4, out_c, kernel_size=1, stride=1, groups=g)

        self.identity = nn.Sequential(
            nn.MaxPool2d(2, 2) if down_sample else nn.Identity(),
            nn.Conv2d(in_c, out_c, 1, 1, groups=g) if in_c != out_c else nn.Identity(),
        )

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        return x


class ConvNeXtV2Blk(nn.Module):
    def __init__(self, dim, act=nn.GELU):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = act()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Network(nn.Module):
    def __init__(
            self,
            num_class,
            img_size=224,
            network_type: int = 0,
            use_pretrained_resnet: bool = False,
    ):
        super(Network, self).__init__()
        activation = nn.GELU
        glo_pool = nn.AdaptiveAvgPool2d(1)
        flat = nn.Flatten()
        self.network_type = network_type
        self.img_size = img_size

        if use_pretrained_resnet:
            encoder = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            modules = list(encoder.children())[:-3]
            self.encoder = nn.Sequential(
                torch.nn.Sequential(*modules),  # [B, 1024, 14, 14]
                nn.Conv2d(1024, 512, kernel_size=1, stride=1),
                nn.BatchNorm2d(512),
                activation(),
            )
        elif network_type < 2:
            grouped = network_type == 1
            self.encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    ResNetBlk(64, 64, down_sample=False, group=grouped),
                ),
                nn.Sequential(
                    ResNetBlk(64, 128, down_sample=True, group=grouped),
                    ResNetBlk(128, 128, down_sample=False, group=grouped),
                ),
                nn.Sequential(
                    ResNetBlk(128, 256, down_sample=True, group=grouped),
                    ResNetBlk(256, 256, down_sample=False, group=grouped),
                ),
                nn.Sequential(
                    ResNetBlk(256, 512, down_sample=True, group=grouped),
                    ResNetBlk(512, 512, down_sample=False, group=grouped),
                ),
            )
        elif network_type == 2:
            # ConvNextv2 alike
            self.encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    ConvNeXtV2Blk(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(128, data_format='channels_first'),
                    ConvNeXtV2Blk(128),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(256, data_format='channels_first'),
                    ConvNeXtV2Blk(256),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(512, data_format='channels_first'),
                ),
            )
        else:
            if network_type == 3:
                block = lambda dim: partial(ResNetBlk, act=activation, group=True, down_sample=False)(dim, dim)
            else:
                block = partial(ConvNeXtV2Blk, act=activation)
            # Image transformer + ConvNextv2
            self.encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=5, bias=False),
                    nn.BatchNorm2d(64),
                    activation(),
                    block(64),
                    block(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                    nn.MaxPool2d(2, 2),
                    nn.BatchNorm2d(128),
                    activation(),
                    block(128),
                    block(128),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=1, groups=8),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(256, data_format='channels_first'),
                    activation(),
                    block(256),
                ),
                nn.Sequential(
                    LayerNorm(256, data_format='channels_first'),
                    ViTEncoderLayer(img_size // 8, 7, 256, 128, 4, batch_first=True),
                    nn.Linear(128, 256),
                    nn.LayerNorm(256),
                    activation(),
                ),
                nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LayerNorm(512),
                    activation(),
                )
            )

        if num_class == 2:
            num_class -= 1
        self.decoder = nn.Sequential(
            glo_pool,
            flat,
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, num_class),
        )
        self.mae = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # encoder part
        x = self.encoder(x)
        if self.network_type >= 3 and self.network_type != 5:
            xs = x.reshape(x.shape[0], 7, 7, -1).permute(0, 3, 1, 2)
            reconstructed_img = self.mae(xs)
            # for vision transformer, get the last sequence output
            x = x[:, -1, :]
            x = x.reshape(x.shape[0], -1, 1, 1)
        else:
            reconstructed_img = self.mae(x)
        x = self.decoder(x)
        return x, reconstructed_img


if __name__ == '__main__':
    from torchinfo import summary
    for i in range(6):
        if i < 5:
            continue
        print('Network type:', i)
        pretrained_res50 = i == 5
        network = Network(2, network_type=i, img_size=224, use_pretrained_resnet=pretrained_res50)
        random_in = torch.rand((32, 3, 224, 224))
        output = network(random_in)
        summary(network, input_data=random_in)
        print('\n\n')
