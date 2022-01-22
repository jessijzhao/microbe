import torch

from typing import Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reset_peak_memory_stats(device: torch.device) -> Tuple[int, int]:
    """Safely resets CUDA peak memory statistics of device if it is
    a CUDA device.

    Notes: ``torch.cuda.reset_peak_memory_stats(device)`` will error
    if no CUDA memory has been allocated to the device.

    Args:
        device: A torch.device

    Returns:
        max_memory_allocated before resetting the statistics and
        memory_allocated, both in bytes
    """
    if torch.cuda.is_available():
        prev_max_memory = torch.cuda.max_memory_allocated(device)
        memory_allocated = torch.cuda.memory_allocated(device)

        if prev_max_memory != memory_allocated and prev_max_memory > 0:
            # raises RuntimeError if no previous allocation occurred
            torch.cuda.reset_peak_memory_stats(device)
            assert(torch.cuda.max_memory_allocated(device) == memory_allocated)

        return prev_max_memory, memory_allocated


def get_layer_set(layer: str) -> str:
    """Layers in the same layer set share a config.

    Args:
        layer: Full name of the layer. This will be the PyTorch or Opacus
        name of the layer in lower case (e.g. linear, rnn, dprnn), prefixed with
        gsm_ (e.g. gsm_linear, gsm_dprnn) if DP is enabled. MultiheadAttention
        is abbreviated to mha.

    Returns:
        The name of the layer set, where a set of layers are defined as layers
        that share the same __init__ signature.

    Notes:
        All RNN-based models share a config.

    """
    layer_set = layer.replace("gsm_dp", "").replace("gsm_", "").replace("dp", "")

    # all RNN-based model use the same config
    if layer_set in ['rnn', 'gru', 'lstm']:
        layer_set = 'rnn_base'

    return layer_set
