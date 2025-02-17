from typing import Literal

import mlx.core as mx
import mlx.nn as nn


from .modules import ConvBNLayer


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        num_layers: int = 1,
        dropout: float = 0,
        direction: str = "bidirectional",
        with_linear: bool = False,
    ):
        super().__init__()
        from .rnn import LSTM

        self.with_linear = with_linear
        self.rnn = LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            # dropout=dropout,
            direction=direction,
        )

        if self.with_linear:
            self.linear = nn.Linear(hidden_size * 2, output_size)

    def __call__(self, input_feature: mx.array) -> mx.array:
        recurrent, _ = self.rnn(input_feature)
        if self.with_linear:
            output = self.linear(recurrent)
            return output
        return recurrent


class EncoderWithCascadeRNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: list,
        num_layers: int = 2,
        with_linear: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels[-1]
        self.encoder = [
            BidirectionalLSTM(
                in_channels if i == 0 else out_channels[i - 1],
                hidden_size,
                output_size=out_channels[i],
                num_layers=1,
                with_linear=with_linear,
            )
            for i in range(num_layers)
        ]

    def forward(self, x: mx.array) -> mx.array:
        for l in self.encoder:
            x = l(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def drop_path(x: mx.array, drop_prob: float = 0.0, training: bool = False) -> mx.array:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + mx.random.uniform(shape, dtype=x.dtype)
    random_tensor = mx.floor(random_tensor)
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        return drop_path(x, self.drop_prob, training)


class ConvMixer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, HW: list = [8, 25], local_k: list = [3, 3]):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1,
            [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mixer: Literal["Global", "Local"] = "Global",
        HW: list = None,
        local_k: list = [7, 11],
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = mx.ones([H * W, H + hk - 1, W + wk - 1], dtype=mx.float32)
            for h in range(0, H):
                for w in range(W):
                    mask[h * W + w, h : h + hk, w : w + wk] = 0.0
            mask_mx = mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2].reshape(1, 1, -1)
            mask_inf = mx.full([H * W, H * W], -mx.inf, dtype=mx.float32)
            mask = mx.where(mask_mx < 1, mask_mx, mask_inf)
            self.mask = mask.reshape(1, 1, H * W, H * W)
        else:
            self.mask = None
        self.mixer = mixer

    def __call__(self, x: mx.array) -> mx.array:
        # qkv = self.qkv(x).reshape(0, -1, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        bsz = x.shape[0]
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz, -1, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        # 3, bsz, num_heads, N, head_dim
        q, k, v = qkv
        sdpa = mx.fast.scaled_dot_product_attention
        attn = sdpa(q, k, v, scale=self.scale, mask=self.mask)
        x = attn.transpose(0, 2, 1, 3).reshape(bsz, -1, self.dim)
        return self.proj_drop(self.proj(x))


class SVTRBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: Literal["Global", "Local", "Conv"],
        local_mixer: list = [7, 11],
        HW: int = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: str = "nn.LayerNorm",
        eps: float = 1e-6,
        prenorm: bool = True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=eps)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == "Global" or mixer == "Local":
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "Conv":
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError(f"The mixer must be one of [Global, Local, Conv], got {mixer}")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=eps)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def __call__(self, x: mx.array) -> mx.array:

        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dims: int = 64,
        depth: int = 2,
        hidden_dims: int = 120,
        use_guide: bool = False,
        num_heads: int = 8,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path: float = 0.0,
        kernel_size: list = [3, 3],
        qk_scale: float = None,
    ):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act=nn.SiLU(),
        )
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act=nn.SiLU())

        self.svtr_block = [
            SVTRBlock(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer="Global",
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.SiLU,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer="nn.LayerNorm",
                eps=1e-05,
                prenorm=False,
            )
            for i in range(depth)
        ]

        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act=nn.SiLU())
        self.conv4 = ConvBNLayer(
            2 * in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act=nn.SiLU(),
        )
        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act=nn.SiLU())
        self.out_channels = dims

    def __call__(self, x: mx.array) -> mx.array:
        z = x
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, H, W, C = z.shape
        # z = z.flatten(2).transpose([0, 2, 1])
        z = z.flatten(1, 2)

        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.reshape(-1, H, W, C)
        z = self.conv3(z)
        z = mx.concat((h, z), axis=-1)
        # z = self.conv1x1(self.conv4(z))
        z = self.conv4(z)
        z = self.conv1x1(z)
        return z


class Im2Seq(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        assert H == 1, f"Height must be 1, got {H}"
        x = x.reshape([B, W, C])
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 48):
        super().__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(in_channels, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc(x)


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 48):
        super().__init__()
        from .rnn import LSTM

        self.out_channels = hidden_size * 2
        self.lstm = LSTM(in_channels, hidden_size, direction="bidirectional", num_layers=2)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lstm(x)[0]


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels: int, encoder_type: str, hidden_size: int = 48, **kwargs):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {
                "reshape": Im2Seq,
                "fc": EncoderWithFC,
                "rnn": EncoderWithRNN,
                "svtr": EncoderWithSVTR,
                "cascadernn": EncoderWithCascadeRNN,
            }
            assert encoder_type in support_encoder_dict, f"{encoder_type} must in {support_encoder_dict.keys()}"
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, **kwargs)
            elif encoder_type == "cascadernn":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size, **kwargs
                )
            else:
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def __call__(self, x: mx.array) -> mx.array:
        if self.encoder_type != "svtr":
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x
