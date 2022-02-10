import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_n_byte_tensor(n: int, device: torch.device = device):
    """Returns tensor with n bytes, since int8 = 1 byte."""
    return torch.zeros(n, dtype=torch.int8, device=device)


def get_actual_memory_allocated(n: int, device: torch.device = device):
    """Returns the number of bytes actually allocated for an n byte tensor.
    Reset memory statistics.
    """
    prev_memory_allocated = torch.cuda.memory_allocated(device)
    tensor = get_n_byte_tensor(n)
    memory_allocated = torch.cuda.memory_allocated(device)
    del tensor
    torch.cuda.reset_peak_memory_stats(device)
    assert prev_memory_allocated == torch.cuda.memory_allocated(device)
    return memory_allocated - prev_memory_allocated
