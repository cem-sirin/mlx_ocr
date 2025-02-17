import math
from typing import Optional, Tuple, Literal

import mlx.core as mx
import mlx.nn as nn

import numpy as np


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        direction: Literal["forward", "bidirectional"] = "forward",
        bias: bool = True,
    ):
        super().__init__()
        self.direction = direction
        self.layers = []

        layer_cls = RNN if direction == "forward" else BiRNN
        for i in range(num_layers):
            self.layers.append(layer_cls("lstm", input_size, hidden_size, bias))
            input_size = hidden_size

    def __call__(self, x: mx.array, hidden: Optional[mx.array] = None, cell: Optional[mx.array] = None):
        hidden, cell = self._validate_hc(hidden, cell)

        for layer in self.layers:
            # x, states = layer(x, hidden, cell)
            x, states = layer(x, None, None)
            hidden, cell = states
        return x, (hidden, cell)

    def _validate_hc(self, hidden: Optional[mx.array], cell: Optional[mx.array]):
        if self.direction == "bidirectional" and hidden is None:
            hidden = (None, None)
        if self.direction == "bidirectional" and cell is None:
            cell = (None, None)

        return hidden, cell


class RNN(nn.Module):
    def __init__(self, cell_type: Literal["lstm"], input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        if cell_type == "lstm":
            self.cell = LSTMCell(input_size, hidden_size, bias)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

    def __call__(
        self, x: mx.array, hidden: Optional[mx.array] = None, cell: Optional[mx.array] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        seq_len = x.shape[-2]
        output = []
        for t in range(seq_len):
            x_t = x[:, t]
            x_t, (hidden, cell) = self.cell(x_t, hidden, cell)  # unpack states tuple
            output.append(x_t)

        output = mx.stack(output, axis=-2)
        return output, (hidden, cell)


class BiRNN(nn.Module):
    def __init__(self, cell_type: Literal["lstm"], input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        if cell_type == "lstm":
            self.cell_fw = LSTMCell(input_size, hidden_size, bias)
            self.cell_bw = LSTMCell(input_size, hidden_size, bias)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

        self.hidden_size: int = self.cell_fw.hidden_size  # Assuming both cells have the same hidden size

    def __call__(
        self,
        x: mx.array,
        hidden: Optional[tuple[mx.array, mx.array]] = (None, None),
        cell: Optional[tuple[mx.array, mx.array]] = (None, None),
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Unpack hidden and cell states tuple
        hidden_fw, hidden_bw = hidden
        cell_fw, cell_bw = cell
        seq_len = x.shape[0]

        # Forward direction
        output_fw = []
        for t in range(seq_len):
            x_t, (hidden_fw, cell_fw) = self.cell_fw(x[t], hidden_fw, cell_fw)
            output_fw.append(x_t)
        output_fw = mx.stack(output_fw)

        # Backward direction
        output_bw = []
        for t in reversed(range(seq_len)):
            x_t, (hidden_bw, cell_bw) = self.cell_bw(x[t], hidden_bw, cell_bw)
            output_bw.append(x_t)
        output_bw = mx.stack(output_bw[::-1])

        # output = mx.concatenate([output_fw, output_bw], axis=-1)
        output = output_fw + output_bw

        hidden = (hidden_fw, hidden_bw)
        cell = (cell_fw, cell_bw)
        return output, (hidden, cell)


class LSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        scale = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size, input_size))
        self.weight_hh = mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size, hidden_size))
        self.bias_ih = mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,)) if bias else None
        self.bias_hh = mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,)) if bias else None

    def __repr__(self):
        return f"LSTMCell({self.input_size}, {self.hidden_size}, bias={self.bias})"

    def __call__(
        self,
        x: mx.array,
        hidden: Optional[mx.array] = None,
        cell: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:

        if hidden is None:
            hidden = mx.zeros(self.hidden_size)
        if cell is None:
            cell = mx.zeros(self.hidden_size)

        if self.bias_ih is not None:
            z = mx.addmm(self.bias_ih, x, self.weight_ih.T)
        else:
            z = x @ self.weight_ih.T

        if self.bias_hh is not None:
            ifgo = mx.addmm(self.bias_hh, hidden, self.weight_hh.T) + z
        else:
            ifgo = hidden @ self.weight_hh.T + z

        i, f, g, o = mx.split(ifgo, 4, axis=-1)
        i = mx.sigmoid(i)
        f = mx.sigmoid(f)
        g = mx.tanh(g)
        o = mx.sigmoid(o)

        new_cell = f * cell + i * g
        new_hidden = o * mx.tanh(new_cell)

        return new_hidden, (new_hidden, new_cell)
