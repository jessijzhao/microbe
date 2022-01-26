import torch
import pytest

from utils import device, get_layer_set, reset_peak_memory_stats
from utils_test import get_n_byte_tensor, get_actual_memory_allocated

@pytest.mark.parametrize('layer_set, layers', [
    # Pytorch layers are named layer (no DP) or gsm_layer (DP)
    ('linear', ['linear', 'gsm_linear']),
    ('conv', ['conv', 'gsm_conv']),
    # Opacus layers are named dplayer (no DP) or gsm_dplayer (DP)
    ('mha', ['mha', 'dpmha', 'gsm_dpmha']),
    # RNN-based models share the same interface
    ('rnn_base', ['rnn', 'dprnn', 'gsm_dprnn']),
    ('rnn_base', ['lstm', 'dplstm', 'gsm_dplstm'])
])
def test_get_layer_set(layer_set, layers):
    assert(all(get_layer_set(layer) == layer_set for layer in layers))


@pytest.mark.parametrize('prev_max_memory, allocated_memory', [
    # prev_max_memory = allocated_memory = 0 --> (0, 0)
    (0, 0),
    # prev_max_memory = allocated_memory > 0 --> (prev_max_memory, prev_max_memory)
    (1, 1),
    # prev_max_memory > allocated_memory = 0 --> (prev_max_memory, 0)
    (1, 0),
    # prev_max_memory > allocated_memory > 0 --> (prev_max_memory, allocated_memory)
    (2, 1),
])
def test_reset_peak_memory_stats(prev_max_memory: int, allocated_memory: int):
    # keep x, delete y
    x = get_n_byte_tensor(allocated_memory)
    y = get_n_byte_tensor(prev_max_memory - allocated_memory)
    del y

    # get the true allocated memory (CUDA memory is allocated in blocks)
    prev_max_memory = torch.cuda.max_memory_allocated(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    assert(reset_peak_memory_stats(device) == (prev_max_memory, allocated_memory))

    # clean up
    del x
    assert(reset_peak_memory_stats(device) == (allocated_memory, 0))



