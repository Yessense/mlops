from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
    ):
        self.patch_size = (patch_size, patch_size)
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(
                in_channels,
                emb_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        img_size = (img_size, img_size)
        self.num_patches = (img_size[1] // self.patch_size[1]) * (
            img_size[0] // self.patch_size[0]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(
            torch.randn((img_size[0] // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        #################
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        ################

        #
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim, num_heads, mlp_ratio, drop_rate) for i in range(depth)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            emb_size=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformer = Transformer(
            depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x):
        # Часть 1
        x = self.patch_embed(x)

        # Часть 2
        x = self.transformer(x)
        x = self.norm(x)

        # Классификация
        x = self.head(x[:, 0])
        return x


class ViTLightningModule(L.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 0.003) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.model = ViT(num_classes=num_classes)
        self.lr = lr
        self.metric = MulticlassAccuracy(num_classes=self.num_classes)

        self.save_hyperparameters()

    def forward(self, x) -> Any:
        return self.model(x)

    def loss(self, y_true, y_pred):
        return F.cross_entropy(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_pred = self(x)

        loss = self.loss(y_true, y_pred)
        self.log("train loss", loss)
        self.log("train acc", self.metric(y_pred, y_true))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch

        y_pred = self(x)

        loss = self.loss(y_true, y_pred)
        self.log("val loss", loss)
        self.log("val acc", self.metric(y_pred, y_true))

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    net = ViTLightningModule()
    img = torch.randn(1, 3, 224, 224)

    out = net(img)
    print("Done")
