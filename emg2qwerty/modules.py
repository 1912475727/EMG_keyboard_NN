# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


# -----------------------------------------------------------------------------
# CNN + Transformer: simple two-stage encoder (local features then global context).
# Good starting point before moving to full Conformer.
# -----------------------------------------------------------------------------


class TemporalCNNEncoder(nn.Module):
    """Simple stack of 1D temporal convolutions. Preserves sequence length (same padding).
    Input and output shape: (T, N, d_model).

    Args:
        d_model (int): Model dimension (in and out channels).
        n_layers (int): Number of Conv1d layers.
        kernel_size (int): Kernel size for temporal conv (use odd for same padding).
        dropout (float): Dropout after each conv block.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 3,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for same padding"
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    _TemporalConv1dBlock(d_model, kernel_size, padding),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = x + layer(x)  # residual
        return x  # (T, N, d_model)


class _TemporalConv1dBlock(nn.Module):
    """Single Conv1d over time: (T, N, C) -> (T, N, C) with same padding."""

    def __init__(self, d_model: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=1)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, C) -> (N, C, T)
        x = inputs.permute(1, 2, 0)
        x = self.conv(x)
        x = self.activation(x)
        return x.permute(2, 0, 1)  # (T, N, C)


class CNNTransformerEncoder(nn.Module):
    """Two-stage encoder: temporal CNN (local features) then Transformer (global context).
    Input and output shape: (T, N, d_model). No temporal downsampling; suitable for CTC.

    Args:
        d_model (int): Model dimension (e.g. 768).
        cnn_layers (int): Number of CNN layers.
        cnn_kernel_size (int): Kernel size for CNN.
        transformer_layers (int): Number of Transformer encoder layers.
        n_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward hidden dim (often 4 * d_model).
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        cnn_layers: int = 3,
        cnn_kernel_size: int = 31,
        transformer_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model

        self.cnn = TemporalCNNEncoder(
            d_model=d_model,
            n_layers=cnn_layers,
            kernel_size=cnn_kernel_size,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # (T, N, C)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(inputs)  # (T, N, d_model)
        x = self.transformer(x)  # (T, N, d_model)
        return x


# -----------------------------------------------------------------------------
# Conformer: Convolution-augmented Transformer for sequence modeling.
# Reference: Gulati et al., "Conformer: Convolution-augmented Transformer for
# Speech Recognition", https://arxiv.org/abs/2005.08100
# -----------------------------------------------------------------------------


class ConformerConvModule(nn.Module):
    """Convolution module used inside a Conformer block. Applies pre-norm,
    pointwise conv, gated linear unit (GLU), depthwise temporal conv, batch norm,
    Swish, and pointwise conv. Does not apply residual (caller adds it).
    Input and output shape: (T, N, d_model).

    Args:
        d_model (int): Model dimension (number of channels).
        kernel_size (int): Kernel size of the depthwise temporal convolution.
    """

    def __init__(self, d_model: int, kernel_size: int = 31) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for same padding"
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Pre-norm, then (T, N, C) -> (N, C, T) for Conv1d
        x = self.layer_norm(inputs).permute(1, 2, 0)

        # Pointwise + GLU: (N, 2C, T) -> gate and value, (N, C, T)
        x = self.pointwise_1(x)
        x = x.chunk(2, dim=1)[0] * torch.sigmoid(x.chunk(2, dim=1)[1])

        # Depthwise conv (same padding)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_2(x)

        # (N, C, T) -> (T, N, C). Residual is applied in ConformerBlock.
        return x.permute(2, 0, 1)


class ConformerFeedForward(nn.Module):
    """Macaron-style half-step feed-forward. FFN(x) = Linear2(Swish(Linear1(x))).
    Used as 0.5 * FFN(x) in the Conformer block. Input and output: (T, N, d_model).

    Args:
        d_model (int): Model dimension.
        expansion (int): Expansion factor for the hidden layer; hidden = d_model * expansion.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.linear_1 = nn.Linear(d_model, hidden)
        self.linear_2 = nn.Linear(hidden, d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(inputs)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: half FFN -> multi-head self-attention -> conv -> half FFN,
    each with pre-norm and residual. Input and output shape: (T, N, d_model).

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        ff_expansion (int): Feed-forward expansion factor.
        conv_kernel_size (int): Kernel size for the conv module.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn_1 = ConformerFeedForward(d_model, expansion=ff_expansion, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel_size)
        self.norm_3 = nn.LayerNorm(d_model)
        self.ffn_2 = ConformerFeedForward(d_model, expansion=ff_expansion, dropout=dropout)
        self.norm_4 = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Half-step FFN (macaron)
        x = inputs + 0.5 * self.ffn_1(self.norm_1(inputs))

        # Multi-head self-attention (pre-norm, residual)
        attn_in = self.norm_2(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out

        # Convolution module (residual at block level)
        x = x + self.conv_module(self.norm_3(x))

        # Half-step FFN (macaron)
        x = x + 0.5 * self.ffn_2(self.norm_4(x))
        return x


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks. Replaces TDSConvEncoder for sequence modeling.
    Input and output shape: (T, N, d_model). No temporal downsampling.

    Args:
        d_model (int): Model dimension (e.g. 768 to match TDS front-end output).
        n_layers (int): Number of Conformer blocks.
        n_heads (int): Number of attention heads per block.
        ff_expansion (int): Feed-forward expansion factor.
        conv_kernel_size (int): Kernel size in the conv module.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_expansion=ff_expansion,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x  # (T, N, d_model)
