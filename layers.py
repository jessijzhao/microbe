from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from opacus.layers import DPGRU, DPLSTM, DPRNN, DPMultiheadAttention

from utils import reset_peak_memory_stats


class LayerType:
    LINEAR: str = "linear"
    GSM_LINEAR: str = "gsm_linear"
    CONV: str = "conv"
    GSM_CONV: str = "gsm_conv"
    LAYERNORM: str = "layernorm"
    GSM_LAYERNORM: str = "gsm_layernorm"
    INSTANCENORM: str = "instancenorm"
    GSM_INSTANCENORM: str = "gsm_instancenorm"
    GROUPNORM: str = "groupnorm"
    GSM_GROUPNORM: str = "gsm_groupnorm"
    EMBEDDING: str = "embedding"
    GSM_EMBEDDING: str = "gsm_embedding"
    MHA: str = "mha"
    DPMHA: str = "dpmha"
    GSM_DPMHA: str = "gsm_dpmha"
    RNN: str = "rnn"
    DPRNN: str = "dprnn"
    GSM_DPRNN: str = "gsm_dprnn"
    GRU: str = "gru"
    DPGRU: str = "dpgru"
    GSM_DPGRU: str = "gsm_dpgru"
    LSTM: str = "lstm"
    DPLSTM: str = "dplstm"
    GSM_DPLSTM: str = "gsm_dplstm"


class LayerFactory:
    class Layer:
        @abstractmethod
        def __init__(
            self,
            *,
            batch_size: int,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
            **kwargs,
        ):
            if random_seed is not None:
                torch.manual_seed(random_seed)

            self._criterion: Callable = criterion
            self._layer: nn.Module = nn.Module()
            self._layer_inputs: List[Any] = [torch.zeros(batch_size)]
            self._labels: torch.Tensor = torch.zeros(batch_size)

        def to(
            self, device: torch.device, forward_only: bool = False
        ) -> Dict[str, int]:
            assert reset_peak_memory_stats(device).cur_mem == 0

            res = {}

            self._layer = self._layer.to(device)
            res["layer"] = torch.cuda.memory_allocated(device)

            self._layer_inputs = [item.to(device) for item in self._layer_inputs]
            res["inputs"] = torch.cuda.memory_allocated(device) - res["layer"]

            if forward_only:
                self._layer.eval()
            else:
                self._labels = self._labels.to(device)
                res["labels"] = (
                    torch.cuda.memory_allocated(device) - res["layer"] - res["inputs"]
                )

                self._layer.train()

            return res

        def forward_only(self) -> torch.Tensor:
            return self._layer(*self._layer_inputs)

        def forward_backward(self) -> None:
            preds = self.forward_only()
            loss = self._criterion(preds, self._labels)
            loss.backward()
            self._layer.zero_grad()

        def make_private(self) -> None:
            self._layer = GradSampleModule(self._layer)

    class LinearBase(Layer):
        def __init__(
            self,
            *,
            batch_size: int,
            input_shape: Tuple[int, ...],
            in_features: int,
            out_features: int,
            bias: bool = True,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            self._input_tensor = torch.randn(batch_size, *input_shape, in_features)
            self._layer_inputs = [self._input_tensor]
            self._layer = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
            )
            self._labels = torch.randn(batch_size, *input_shape, out_features)
            self._criterion = criterion

    class ConvBase(Layer):
        def __init__(
            self,
            *,
            batch_size: int,
            in_channels: int,
            input_shape: Tuple[int, ...],
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]],
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Union[int, Tuple[int, ...]] = 0,
            dilation: Union[int, Tuple[int, ...]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            D = len(input_shape)
            if D == 1:
                self._layer_name = nn.Conv1d
            elif D == 2:
                self._layer_name = nn.Conv2d
            elif D == 3:
                self._layer_name = nn.Conv3d
            else:
                raise Exception("Input shape must be between 1 and 3 long")

            self._input_tensor = torch.randn(batch_size, in_channels, *input_shape)
            self._layer_inputs = [self._input_tensor]
            self._layer = self._layer_name(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
            outputs = self._layer(self._input_tensor)
            self._labels = torch.randn(outputs.shape)
            del outputs
            self._criterion = criterion

    class LayerNormBase(Layer):
        def __init__(
            self,
            *,
            batch_size: int,
            input_shape: Tuple[int, ...],
            D: int,
            eps: float = 1e-05,
            elementwise_affine: bool = True,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            self._input_tensor = torch.randn(batch_size, *input_shape)
            self._layer_inputs = [self._input_tensor]
            self._layer = nn.LayerNorm(
                normalized_shape=self._input_tensor.shape[-D:],
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
            self._labels = torch.randn(self._input_tensor.shape)
            self._criterion = criterion

    class InstanceNormBase(Layer):
        def __init__(
            self,
            *,
            batch_size: int,
            num_features: int,
            input_shape: Tuple[int, ...],
            eps: float = 1e-05,
            affine: bool = False,
            track_running_stats: bool = False,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            D = len(input_shape)
            if D == 1:
                self._layer_name = nn.InstanceNorm1d
            elif D == 2:
                self._layer_name = nn.InstanceNorm2d
            elif D == 3:
                self._layer_name = nn.InstanceNorm3d
            else:
                raise Exception("Input shape must be between 1 and 3 long")

            self._input_tensor = torch.randn(batch_size, num_features, *input_shape)
            self._layer_inputs = [self._input_tensor]
            self._layer = self._layer_name(
                num_features=num_features,
                eps=eps,
                affine=affine,
                track_running_stats=track_running_stats,
            )
            self._labels = torch.randn(self._input_tensor.shape)
            self._criterion = criterion

    class GroupNormBase(Layer):
        def __init__(
            self,
            *,
            batch_size: int,
            input_shape: Tuple[int, ...],
            num_groups: int,
            num_channels: int,
            eps: float = 1e-05,
            affine: bool = True,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            self._input_tensor = torch.randn(batch_size, num_channels, *input_shape)
            self._layer_inputs = [self._input_tensor]
            self._layer = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine
            )
            self._labels = torch.randn(self._input_tensor.shape)
            self._criterion = criterion

    class EmbeddingBase(Layer):
        def __init__(
            self,
            *,
            batch_size: int,
            input_shape: Tuple[int, ...],
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
            max_norm: Optional[float] = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            self._input_tensor = torch.randint(
                high=num_embeddings,
                size=(batch_size, *input_shape),
                dtype=torch.long,
            )
            self._layer_inputs = [self._input_tensor]
            self._layer = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse,
            )
            self._labels = torch.randn(batch_size, *input_shape, embedding_dim)
            self._criterion = criterion

    class CLayer(Layer):
        """Some layers return multiple tensors."""

        def forward_only(self) -> torch.Tensor:
            return self._layer(*self._layer_inputs)[0]

    class MHABase(CLayer):
        def __init__(
            self,
            *,
            layer: nn.Module,
            batch_size: int,
            source_seq_len: int,
            targ_seq_len: int,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: Optional[int] = None,
            vdim: Optional[int] = None,
            batch_first: bool = False,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            kdim = kdim if kdim else embed_dim
            vdim = vdim if vdim else embed_dim

            self._input_tensor = (
                torch.randn(targ_seq_len, batch_size, embed_dim)
                if not batch_first
                else torch.randn(batch_size, targ_seq_len, embed_dim)
            )
            self._key = (
                torch.randn(source_seq_len, batch_size, kdim)
                if not batch_first
                else torch.randn(batch_size, source_seq_len, kdim)
            )
            self._value = (
                torch.randn(source_seq_len, batch_size, vdim)
                if not batch_first
                else torch.randn(batch_size, source_seq_len, vdim)
            )
            self._layer_inputs = [self._input_tensor, self._key, self._value]

            self._layer = layer(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                kdim=kdim,
                vdim=vdim,
            )
            self._labels = (
                torch.randn(targ_seq_len, batch_size, embed_dim)
                if not batch_first
                else torch.randn(batch_size, targ_seq_len, embed_dim)
            )
            self._criterion = criterion

    class RNNBase(CLayer):
        def __init__(
            self,
            *,
            layer: nn.Module,
            batch_size: int,
            seq_len: int,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = False,
            batch_first: bool = False,
            dropout: float = 0,
            bidirectional: bool = False,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
            **kwargs,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            self._input_tensor = (
                torch.randn(
                    seq_len,
                    batch_size,
                    input_size,
                )
                if not batch_first
                else torch.randn(batch_size, seq_len, input_size)
            )

            D = 2 if bidirectional else 1
            self._h_0 = torch.randn(D * num_layers, batch_size, hidden_size)
            self._layer_inputs = [self._input_tensor, self._h_0]

            self._layer = layer(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
                **kwargs,
            )

            self._labels = (
                torch.randn(seq_len, batch_size, D * hidden_size)
                if not batch_first
                else torch.randn(batch_size, seq_len, D * hidden_size)
            )
            self._criterion = criterion

    class LSTMBase(RNNBase):
        def __init__(
            self,
            *,
            layer: nn.Module,
            batch_size: int,
            seq_len: int,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = False,
            batch_first: bool = False,
            dropout: float = 0,
            bidirectional: bool = False,
            proj_size: int = 0,
            random_seed: Optional[int] = None,
            criterion: Callable = F.cross_entropy,
        ) -> None:
            if random_seed is not None:
                torch.manual_seed(random_seed)

            super().__init__(
                layer=layer,
                batch_size=batch_size,
                seq_len=seq_len,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
                proj_size=proj_size,
            )
            h_out = proj_size if proj_size > 0 else hidden_size
            D = 2 if bidirectional else 1
            self._h_0 = torch.randn(D * num_layers, batch_size, h_out)
            self._c_0 = torch.randn(D * num_layers, batch_size, hidden_size)
            self._layer_inputs = [self._input_tensor, (self._h_0, self._c_0)]

            self._labels = (
                torch.randn(seq_len, batch_size, D * h_out)
                if not batch_first
                else torch.randn(batch_size, seq_len, D * h_out)
            )
            self._criterion = criterion

    @staticmethod
    def make_private(layer: Layer) -> Layer:
        layer.make_private()
        return layer

    @staticmethod
    def create(layer_name: str, **kwargs) -> Layer:
        if layer_name == LayerType.LINEAR:
            return LayerFactory.LinearBase(**kwargs)
        elif layer_name == LayerType.GSM_LINEAR:
            return LayerFactory.make_private(LayerFactory.LinearBase(**kwargs))
        elif layer_name == LayerType.CONV:
            return LayerFactory.ConvBase(**kwargs)
        elif layer_name == LayerType.GSM_CONV:
            return LayerFactory.make_private(LayerFactory.ConvBase(**kwargs))
        elif layer_name == LayerType.LAYERNORM:
            return LayerFactory.LayerNormBase(**kwargs)
        elif layer_name == LayerType.GSM_LAYERNORM:
            return LayerFactory.make_private(LayerFactory.LayerNormBase(**kwargs))
        elif layer_name == LayerType.INSTANCENORM:
            return LayerFactory.InstanceNormBase(**kwargs)
        elif layer_name == LayerType.GSM_INSTANCENORM:
            return LayerFactory.make_private(LayerFactory.InstanceNormBase(**kwargs))
        elif layer_name == LayerType.GROUPNORM:
            return LayerFactory.GroupNormBase(**kwargs)
        elif layer_name == LayerType.GSM_GROUPNORM:
            return LayerFactory.make_private(LayerFactory.GroupNormBase(**kwargs))
        elif layer_name == LayerType.EMBEDDING:
            return LayerFactory.EmbeddingBase(**kwargs)
        elif layer_name == LayerType.GSM_EMBEDDING:
            return LayerFactory.make_private(LayerFactory.EmbeddingBase(**kwargs))
        elif layer_name == LayerType.RNN:
            return LayerFactory.RNNBase(layer=nn.RNN, **kwargs)
        elif layer_name == LayerType.DPRNN:
            return LayerFactory.RNNBase(layer=DPRNN, **kwargs)
        elif layer_name == LayerType.GSM_DPRNN:
            return LayerFactory.make_private(
                LayerFactory.RNNBase(layer=DPRNN, **kwargs)
            )
        elif layer_name == LayerType.GRU:
            return LayerFactory.RNNBase(layer=nn.GRU, **kwargs)
        elif layer_name == LayerType.DPGRU:
            return LayerFactory.RNNBase(layer=DPGRU, **kwargs)
        elif layer_name == LayerType.GSM_DPGRU:
            return LayerFactory.make_private(
                LayerFactory.RNNBase(layer=DPGRU, **kwargs)
            )
        elif layer_name == LayerType.LSTM:
            return LayerFactory.LSTMBase(layer=nn.LSTM, **kwargs)
        elif layer_name == LayerType.DPLSTM:
            return LayerFactory.LSTMBase(layer=DPLSTM, **kwargs)
        elif layer_name == LayerType.GSM_DPLSTM:
            return LayerFactory.make_private(
                LayerFactory.LSTMBase(layer=DPLSTM, **kwargs)
            )
        elif layer_name == LayerType.MHA:
            return LayerFactory.MHABase(layer=nn.MultiheadAttention, **kwargs)
        elif layer_name == LayerType.DPMHA:
            return LayerFactory.MHABase(layer=DPMultiheadAttention, **kwargs)
        elif layer_name == LayerType.GSM_DPMHA:
            return LayerFactory.make_private(
                LayerFactory.MHABase(layer=DPMultiheadAttention, **kwargs)
            )
        else:
            raise Exception(f"Invalid layer type: {layer_name}.")
