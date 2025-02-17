"""Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition
ArXiv: https://arxiv.org/abs/1811.00751

x: input feature map from backbone
hf: hollistic features

Notes: The MultiHead class uses SARHead only in the training mode. So the forward_test function
is not used in PP-OCRv3. Thus, the usage of it remains experimental for now.
"""

from dataclasses import dataclass
from typing import List, Optional
from math import ceil

import mlx.core as mx
import mlx.nn as nn

from . import HeadConfig, HeadType
from ..modules.rnn import LSTM


@dataclass
class SARHeadConfig(HeadConfig):
    in_channels: int = 512
    out_channels: int = 100  # This seems to be the number of classes
    enc_dim: int = 512
    max_text_length: int = 25
    enc_bi_rnn: bool = False
    enc_drop_rnn: float = 0.1
    dec_bi_rnn: bool = False
    dec_drop_rnn: float = 0.0
    d_k: int = 512
    pred_dropout: float = 0.1
    pred_concat: bool = True
    head_type: HeadType = "SARHead"


class SARHead(nn.Module):
    """Show, Attend and Read (SAR) head for text recognition."""

    def __init__(self, config: SARHeadConfig):
        super(SARHead, self).__init__()
        self.encoder = SAREncoder(
            bi_rnn=config.enc_bi_rnn,
            drop_rnn=config.enc_drop_rnn,
            d_model=config.in_channels,
            d_enc=config.enc_dim,
        )
        self.decoder = ParallelSARDecoder(
            out_channels=config.out_channels,
            enc_bi_rnn=config.enc_bi_rnn,
            dec_bi_rnn=config.dec_bi_rnn,
            d_model=config.in_channels,
            d_enc=config.enc_dim,
            d_k=config.d_k,
            pred_dropout=config.pred_dropout,
            max_text_length=config.max_text_length,
            pred_concat=config.pred_concat,
        )

    def __call__(self, x: mx.array, labels: Optional[mx.array] = None, valid_ratios: Optional[List[float]] = None):
        """Forward pass of SAR head.

        Args:
            x (mx.array): Input feature map from backbone. (batch_size, height, width, channels)
            labels (mx.array): Encoded labels for SAR head. (batch_size, max_text_length)
            valid_ratios (mx.array): Ratio of the width of the actual image and the padded image. (batch_size,)
        """
        print(f"[SARHead] x.shape {x.shape}")
        hf = self.encoder(x, valid_ratios)  # bsz c
        print(f"[SARHead] hf.shape={hf.shape}")
        return self.decoder(x, hf, labels=labels, valid_ratios=valid_ratios)


class SAREncoder(nn.Module):
    """
    Args:
        bi_rnn (bool): If True, use bidirectional RNN in encoder.
        drop_rnn (float): Dropout probability of RNN layer in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(
        self,
        bi_rnn: bool = False,
        drop_rnn: float = 0.1,
        d_model: int = 512,
        d_enc: int = 512,
        mask: bool = True,
    ):
        super(SAREncoder, self).__init__()

        self.enc_bi_rnn = bi_rnn
        self.enc_drop_rnn = drop_rnn
        self.mask = mask

        # LSTM Encoder
        direction = "bidirectional" if bi_rnn else "forward"
        self.rnn_encoder = LSTM(input_size=d_model, hidden_size=d_enc, num_layers=2, direction=direction)

        # global feature transformation
        encoder_rnn_out_size = d_enc * (int(bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def __call__(self, x: mx.array, valid_ratios: Optional[List[float]] = None) -> mx.array:
        import numpy as np

        # Max pool on the height axis
        x_v = x.max(axis=-3)  # bsz, W, C
        hf = self.rnn_encoder(x_v)[0]  # bsz, W, C or (bsz, T, C)
        bsz, T, C = hf.shape
        if valid_ratios is not None:
            valid_hf = []
            for i in range(len(valid_ratios)):
                valid_width = min(ceil(valid_ratios[i] * T), T)
                valid_hf.append(hf[i, valid_width - 1])
            valid_hf = mx.stack(valid_hf)
        else:
            # Last feature of the sequence
            valid_hf = hf[:, -1, :]  # bsz, C
        hf = self.linear(valid_hf)  # bsz, C
        return hf


class ParallelSARDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,  # 90 + unknown + start + padding
        enc_bi_rnn: bool = False,
        dec_bi_rnn: bool = False,
        d_model: int = 512,
        d_enc: int = 512,
        d_k: int = 64,
        pred_dropout: float = 0.1,
        max_text_length: int = 30,
        mask: bool = True,
        pred_concat: bool = True,
    ):
        super(ParallelSARDecoder, self).__init__()

        self.num_classes = out_channels
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat

        # Encoder and Decoder RNN output size
        enc_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        dec_rnn_out_size = enc_rnn_out_size * (int(dec_bi_rnn) + 1)

        # 2D attention layer
        self.conv1x1_1 = nn.Linear(dec_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)

        # Decoder RNN
        direction = "bidirectional" if dec_bi_rnn else "forward"
        self.rnn_decoder = LSTM(enc_rnn_out_size, enc_rnn_out_size, num_layers=2, direction=direction)

        # Embedding layer
        self.embedding = nn.Embedding(self.num_classes, enc_rnn_out_size)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = self.num_classes - 1

        fc_in_channel = d_model if not pred_concat else dec_rnn_out_size + d_model + enc_rnn_out_size
        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)
        self.enc_rnn_out_size = enc_rnn_out_size

    def _2d_attention(
        self,
        decoder_input: mx.array,
        x: mx.array,
        hf: mx.array,
        valid_ratios: Optional[List[float]] = None,
    ) -> mx.array:
        # TODO: Use fast sdpa
        y = self.rnn_decoder(decoder_input)[0]  # bsz * seq_len * C

        # Query
        q = self.conv1x1_1(y)  # bsz * seq_len * d_k
        q = mx.expand_dims(q, axis=2)  # bsz * seq_len * 1 * d_k
        q = mx.expand_dims(q, axis=3)  # bsz * seq_len * 1 * 1 * d_k

        # Key
        k = self.conv3x3_1(x)  # bsz * H * W * C
        k = mx.expand_dims(k, axis=1)  # bsz * 1 * H * W * d_k

        # Attention weights
        a = mx.tanh(q + k)  # bsz * seq_len * H * W * d_k
        a = self.conv1x1_2(a)  # bsz * seq_len * H * W * 1

        bsz, T, h, w, c = a.shape
        assert c == 1

        # TODO: Change this with a mask
        if valid_ratios is not None:
            for i in range(len(valid_ratios)):
                valid_width = min(w, ceil(valid_ratios[i] * w))
                if valid_width < w:
                    a[i, :, :, valid_width:, :] = float("-inf")

        a = a.flatten(2)  # bsz * seq_len * HW * 1
        a = mx.softmax(a, axis=-1)
        a = a.reshape((bsz, T, h, w, 1))  # bsz * seq_len * H * W * 1

        x = mx.expand_dims(x, axis=1)  # bsz * 1 * H * W * C
        a = mx.sum(x * a, axis=(2, 3))  # bsz * seq_len * C

        if self.pred_concat:
            hf = mx.repeat(hf, repeats=a.shape[1], axis=1)
            a = mx.concat((y, a, hf), axis=-1)

        return self.pred_dropout(self.prediction(a))

    def forward_train(
        self, x: mx.array, hf: mx.array, labels: mx.array, valid_ratios: Optional[List[float]] = None
    ) -> mx.array:
        lab_embedding = self.embedding(labels)  # bsz * seq_len * emb_dim
        hf = mx.expand_dims(hf, axis=1)
        in_dec = mx.concat((hf, lab_embedding), axis=1)
        out_dec = self._2d_attention(in_dec, x, hf, valid_ratios)

        return out_dec[:, 1:, :]  # bsz * seq_len * num_classes

    def _create_start_token(self, bsz: int, seq_len: int) -> mx.array:
        start_token = mx.full((bsz,), vals=self.start_idx, dtype=mx.int64)
        start_token = self.embedding(start_token).reshape(bsz, 1, self.enc_rnn_out_size)
        start_token = mx.repeat(start_token, repeats=seq_len, axis=1)
        return start_token

    def forward_test(self, x: mx.array, hf: mx.array, valid_ratios: Optional[List[float]] = None) -> mx.array:
        seq_len = self.max_seq_len
        bsz = x.shape[0]
        start_token = self._create_start_token(bsz, seq_len)  # bsz * seq_len * emb_dim
        hf = hf.reshape((bsz, 1, self.enc_rnn_out_size))  # bsz * 1 * emb_dim
        decoder_input = mx.concat((hf, start_token), axis=1)  # bsz * (seq_len + 1) * emb_dim

        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(decoder_input, x, hf)  # (bsz, seq_len, num_classes)
            char_output = mx.softmax(decoder_output[:, i, :], -1)  # (bsz, num_classes)

            max_idx = mx.argmax(char_output, axis=-1)  # (bsz,)
            if i < seq_len:
                # I find the "i+1" wrong here, let's see if it works
                decoder_input[:, i + 1, :] = self.embedding(max_idx)

            outputs += [char_output]

        outputs = mx.stack(outputs, 1)  # (bsz, seq_len, num_classes)
        return outputs

    def __call__(
        self,
        x: mx.array,
        hf: mx.array,
        labels: mx.array = None,
        valid_ratios: Optional[List[float]] = None,
    ) -> mx.array:
        if self.training:
            return self.forward_train(x, hf, labels, valid_ratios)
        else:
            return self.forward_test(x, hf, valid_ratios)
