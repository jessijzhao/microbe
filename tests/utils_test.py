import torch

from utils import device, reset_peak_memory_stats


def get_n_byte_tensor(n: int, device: torch.device = device):
    """Returns tensor with n bytes, since int8 = 1 byte."""
    return torch.zeros(n, dtype=torch.int8, device=device)

def get_actual_memory_allocated(n: int, device : torch.device = device):
    """Returns the number of bytes actually allocated for an n byte tensor."""
    assert(reset_peak_memory_stats(device)[1] == 0)
    tensor = get_n_byte_tensor(n)
    del tensor
    max_mem, cur_mem = reset_peak_memory_stats(device)
    assert cur_mem == 0
    return max_mem