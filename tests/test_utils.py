import torch
import unittest

from utils import device, get_layer_set, reset_peak_memory_stats


class UtilsTest(unittest.TestCase):
    
    def test_get_layer_set(self):
        # Pytorch layers are named layer (no DP) or gsm_layer (DP)
        assert(get_layer_set('linear') == get_layer_set('gsm_linear') == 'linear')
        assert(get_layer_set('conv') == get_layer_set('gsm_conv') == 'conv')
        # Opacus layers are named dplayer (no DP) or gsm_dplayer (DP)
        assert(get_layer_set('mha') == get_layer_set('dpmha') == get_layer_set('gsm_dpmha') == 'mha')
        # RNN-based models share the same interface
        assert(get_layer_set('rnn') == get_layer_set('dprnn') == get_layer_set('gsm_dprnn') == 'rnn_base')
        assert(get_layer_set('lstm') == get_layer_set('dplstm') == get_layer_set('gsm_dplstm') == 'rnn_base')
    
    def test_reset_peak_memory_stats(self):
        
        # prev_max_memory = allocated_memory = 0 --> (0, 0)
        prev_max_memory = torch.cuda.max_memory_allocated(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        assert(prev_max_memory == allocated_memory == 0)
        assert(reset_peak_memory_stats(device) == (0, 0))

        # prev_max_memory = allocated_memory > 0 --> (prev_max_memory, prev_max_memory)
        x = torch.zeros(1, device=device)
        prev_max_memory = torch.cuda.max_memory_allocated(device)
        assert(prev_max_memory == torch.cuda.memory_allocated(device) > 0)
        assert(reset_peak_memory_stats(device) == (prev_max_memory, prev_max_memory))

        # prev_max_memory > allocated_memory = 0 --> (prev_max_memory, 0)
        x = torch.zeros(1, device=device)
        del x
        prev_max_memory = torch.cuda.max_memory_allocated(device)
        assert(prev_max_memory > torch.cuda.memory_allocated(device) == 0)
        assert(reset_peak_memory_stats(device) == (prev_max_memory, 0))

        # prev_max_memory > allocated_memory > 0 --> (prev_max_memory, allocated_memory)
        x, y = torch.zeros(1, device=device), torch.ones(1, device=device)
        del x
        prev_max_memory = torch.cuda.max_memory_allocated(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        assert(prev_max_memory > allocated_memory > 0)
        assert(reset_peak_memory_stats(device) == (prev_max_memory, allocated_memory))

        # clean-up
        del y
        assert(reset_peak_memory_stats(device)[1] == 0)



