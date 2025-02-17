from typing import Optional
import mlx.core as mx
import mlx.nn as nn

ACT_DICT = {"relu": nn.ReLU(), "hardswish": nn.Hardswish(), None: nn.Identity()}


class HardSigmoid(nn.Module):
    def __init__(self, slope: float = 1 / 6, offset: float = 0.5):
        super(HardSigmoid, self).__init__()
        self.slope = slope
        self.offset = offset

    def __call__(self, x: mx.array):
        x = x * self.slope + self.offset
        x = mx.minimum(mx.maximum(x, 0), 1)
        return x


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: int = 1):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        return x.mean((1, 2), keepdims=True)


class SEModule(nn.Module):
    """Squeeze-and-Excitation Module."""

    def __init__(self, in_channels: int, reduction: int = 4, slope=0.2):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.hard_sigmoid = HardSigmoid(slope=slope)

    def __call__(self, x: mx.array):
        y = self.avg_pool(x)
        y = self.relu(self.conv1(y))
        y = self.hard_sigmoid(self.conv2(y))
        return x * y


class DSConv(nn.Module):
    """Depthwise Separable Convolution"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        stride=1,
        groups=None,
        bias: bool = False,
        act="relu",
    ):
        super(DSConv, self).__init__()
        assert act in [None, "relu", "hardswish"]  # This is just to match original code
        self.act = ACT_DICT[act]
        if groups == None:
            groups = in_channels

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm(in_channels)

        self.conv2 = nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm(int(in_channels * 4))

        self.conv3 = nn.Conv2d(int(in_channels * 4), out_channels, kernel_size=1, stride=1, bias=bias)
        self._c = [in_channels, out_channels]

        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs):
        x = self.bn1(self.conv1(inputs))
        x = self.bn2(self.conv2(x))
        x = self.act(x)
        x = self.conv3(x)

        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        # Note: I think there should be an else: inputs + x statement here
        # but I am not touching to match the original code

        return x


class IntraCLBlock(nn.Module):
    """Intra Collaborative Learning Block"""

    def __init__(self, in_channels=96, reduce_factor=4):
        super(IntraCLBlock, self).__init__()
        self.channels = in_channels
        self.rf = reduce_factor

        c = self.channels
        rf = self.rf

        self.conv1x1_reduce_channel = nn.Conv2d(c, c // rf, kernel_size=1, stride=1, padding=0)
        self.conv1x1_return_channel = nn.Conv2d(c // rf, c, kernel_size=1, stride=1, padding=0)

        self.v_layer_7x1 = nn.Conv2d(c // rf, c // rf, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))
        self.v_layer_5x1 = nn.Conv2d(c // rf, c // rf, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.v_layer_3x1 = nn.Conv2d(c // rf, c // rf, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.q_layer_1x7 = nn.Conv2d(c // rf, c // rf, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3))
        self.q_layer_1x5 = nn.Conv2d(c // rf, c // rf, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.q_layer_1x3 = nn.Conv2d(c // rf, c // rf, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

        # base
        self.c_layer_7x7 = nn.Conv2d(c // rf, c // rf, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.c_layer_5x5 = nn.Conv2d(c // rf, c // rf, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.c_layer_3x3 = nn.Conv2d(c // rf, c // rf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bn = nn.BatchNorm(c)
        self.relu = nn.ReLU()

    def __call__(self, x: mx.array) -> mx.array:
        x_new = self.conv1x1_reduce_channel(x)

        x_7_c = self.c_layer_7x7(x_new)
        x_7_v = self.v_layer_7x1(x_new)
        x_7_q = self.q_layer_1x7(x_new)
        x_7 = x_7_c + x_7_v + x_7_q

        x_5_c = self.c_layer_5x5(x_7)
        x_5_v = self.v_layer_5x1(x_7)
        x_5_q = self.q_layer_1x5(x_7)
        x_5 = x_5_c + x_5_v + x_5_q

        x_3_c = self.c_layer_3x3(x_5)
        x_3_v = self.v_layer_3x1(x_5)
        x_3_q = self.q_layer_1x3(x_5)
        x_3 = x_3_c + x_3_v + x_3_q

        x_relation = self.conv1x1_return_channel(x_3)

        x_relation = self.bn(x_relation)
        x_relation = self.relu(x_relation)

        return x + x_relation


class ConvBNLayer(nn.Module):
    """Conv2d + BatchNorm + Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: str = None,
    ):
        super(ConvBNLayer, self).__init__()
        # assert act in [None, "relu", "hardswish"]
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm(num_features=out_channels)
        if isinstance(act, str) or act is None:
            self.act = ACT_DICT[act]
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            raise ValueError(f"Invalid activation {act}")

    def __call__(self, x: mx.array) -> mx.array:
        return self.act(self.bn(self.conv(x)))


def make_divisible(v: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Makes the value divisible by the divisor.
    Args:
        v (int): The original value.
        divisor (int): The divisor of the new value.
        min_value (Optional[int]): The minimum value of the new value.

    Returns:
        int: The new value.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
