from typing import Any, Dict, List, Tuple

import pytest
import torch.nn as nn
from opacus.grad_sample import GradSampleModule
from opacus.layers import DPGRU, DPLSTM, DPRNN, DPMultiheadAttention

from layers import LayerFactory


@pytest.mark.parametrize(
    "layers, layer_config",
    [
        (
            [("linear", nn.Linear), ("gsm_linear", nn.Linear)],
            {"input_shape": [], "in_features": 512, "out_features": 512},
        ),
        (
            [("conv", nn.Conv2d), ("gsm_conv", nn.Conv2d)],
            {
                "in_channels": 64,
                "input_shape": [50, 100],
                "out_channels": 64,
                "kernel_size": 8,
            },
        ),
        (
            [("layernorm", nn.LayerNorm), ("gsm_layernorm", nn.LayerNorm)],
            {"input_shape": [64], "D": 1},
        ),
        (
            [
                ("instancenorm", nn.InstanceNorm1d),
                ("gsm_instancenorm", nn.InstanceNorm1d),
            ],
            {"num_features": 256, "input_shape": [64], "affine": True},
        ),
        (
            [("groupnorm", nn.GroupNorm), ("gsm_groupnorm", nn.GroupNorm)],
            {"input_shape": [], "num_groups": 16, "num_channels": 256},
        ),
        (
            [("embedding", nn.Embedding), ("gsm_embedding", nn.Embedding)],
            {"input_shape": [], "num_embeddings": 20000, "embedding_dim": 100},
        ),
        (
            [
                ("mha", nn.MultiheadAttention),
                ("dpmha", DPMultiheadAttention),
                ("gsm_dpmha", DPMultiheadAttention),
            ],
            {
                "source_seq_len": 128,
                "targ_seq_len": 64,
                "embed_dim": 100,
                "num_heads": 4,
            },
        ),
        (
            [
                ("rnn", nn.RNN),
                ("dprnn", DPRNN),
                ("gsm_dprnn", DPRNN),
                ("gru", nn.GRU),
                ("dpgru", DPGRU),
                ("gsm_dpgru", DPGRU),
                ("lstm", nn.LSTM),
                ("dplstm", DPLSTM),
                ("gsm_dplstm", DPLSTM),
            ],
            {"seq_len": 128, "input_size": 100, "hidden_size": 100},
        ),
    ],
)
def test_layers(
    layers: List[Tuple[str, nn.Module]], layer_config: Dict[str, Any]
) -> None:
    """For each supported layer, tests that it is instantiated with the correct module
    and DP support

    Args:
        layers: list of tuples of form (layer_name, module)
        layer_config: config for instantiating the layer
    """
    for layer_name, module in layers:
        layer = LayerFactory.create(
            layer_name=layer_name,
            batch_size=64,
            **layer_config,
        )

    if "gsm" in layer_name:
        assert isinstance(layer._layer, GradSampleModule)
        assert isinstance(layer._layer.to_standard_module(), module)
    else:
        assert isinstance(layer._layer, module)
