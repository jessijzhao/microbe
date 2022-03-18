import math
import time
from typing import Dict, List

import pytest
import torch
import torch.nn as nn
from helpers import get_actual_memory_allocated, get_n_byte_tensor, skipifnocuda

from benchmark_layer import run_layer_benchmark
from layers import LayerFactory
from utils import reset_peak_memory_stats


# number of repeats to test runtime and memory benchmarks for
NUM_REPEATS = [10, 20, 50]


class FakeModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._runtime = kwargs.get("runtime", 0)
        self._pass_memory = kwargs.get("pass_memory", 0)
        self._parameters = {
            "fake_param": get_n_byte_tensor(kwargs.get("layer_memory", 0))
        }


class FakeLayer(LayerFactory.Layer):
    """Fake layer to test runtime and memory benchmarking

    Kwargs:
        runtime: duration for one forward or backward pass
        pass_memory: memory used during a forward or backward pass
        layer_memory: memory taken up by the inputs and layer
    """

    def __init__(self, **kwargs) -> None:
        self._runtime = kwargs.get("runtime", 0)
        self._pass_memory = kwargs.get("pass_memory", 0)
        self._module = FakeModule(**kwargs)

    def to(self, device: torch.device) -> Dict[str, int]:
        self._module = self._module.to(device)
        return {
            "layer": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        }

    def forward_only(self) -> torch.Tensor:
        """Wait for self.duration and allocate self.max_memory bytes"""
        time.sleep(self._runtime)
        tensor = get_n_byte_tensor(
            self._pass_memory,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        return tensor

    def forward_backward(self) -> None:
        """2x runtime and memory of forward_only"""
        _ = self.forward_only()
        _ = self.forward_only()


@pytest.mark.parametrize(
    "duration",
    [(0), (0.005), (0.01), (0.05)],
)
def test_runtime_benchmark(duration: float) -> None:
    """Tests runtime benchmarks on a dummy layer.

    Args:
        duration: duration (s) to test runtime benchmarks for
    """
    for num_repeats in NUM_REPEATS:
        for forward_only in [False, True]:
            runtime, memory_stats = run_layer_benchmark(
                layer_name="",
                batch_size=0,
                num_repeats=num_repeats,
                forward_only=forward_only,
                create_layer=FakeLayer,
                runtime=duration,
            )
            # account for small variations
            assert abs(runtime - ((2 - forward_only) * duration)) < 0.002

            # check that no memory allocation took place
            assert memory_stats["layer"] == 0 and memory_stats["max_memory"] == 0


@skipifnocuda
@pytest.mark.parametrize(
    "pass_bytes_list, layer_bytes_list",
    [([1, 128, 256, 500, 512, 1024, 2000], [0, 128, 500])],
)
def test_memory_benchmark(
    pass_bytes_list: List[int], layer_bytes_list: List[int], strict: bool = True
) -> None:
    """Tests CUDA max_memory benchmarks on a dummy layer.

    Notes: During the experiments included in the paper, CUDA memory is allocated
    in blocks, where block sizes vary across kernels. New CUDA memory is
    allocated for each new tensor. The strict test will fail under a different
    allocation scheme.

    Args:
        pass_bytes_list: list of bytes that each forward or backward pass allocates
        layer_bytes_list: list of bytes that the input and layer allocate
        strict: whether to predict each measurement based on block size
    """
    device = torch.device("cuda:0")

    if strict:
        # find the block size by creating a tensor of size 1 byte
        tiny_tensor = get_n_byte_tensor(1, device=device)
        BLOCK_SIZE = torch.cuda.max_memory_allocated(device)
        del tiny_tensor

    # bytes allocated during forward/backward pass
    for pass_bytes in pass_bytes_list:
        true_pass_memory = get_actual_memory_allocated(pass_bytes, device=device)
        if strict:
            num_blocks_pass = math.ceil(pass_bytes / BLOCK_SIZE)

        # bytes allocated for the layer, inputs, etc.
        for layer_bytes in layer_bytes_list:
            true_layer_memory = get_actual_memory_allocated(layer_bytes, device=device)
            print(layer_bytes, true_layer_memory)

            if strict:
                num_blocks_layer = math.ceil(layer_bytes / BLOCK_SIZE)

            for num_repeats in NUM_REPEATS:
                for forward_only in [False, True]:
                    # reset memory stats and ensure there is no memory leakage
                    assert reset_peak_memory_stats(device).cur_mem == 0

                    runtime, memory_stats = run_layer_benchmark(
                        layer_name="",
                        batch_size=0,
                        num_repeats=num_repeats,
                        forward_only=forward_only,
                        create_layer=FakeLayer,
                        layer_memory=layer_bytes,
                        pass_memory=pass_bytes,
                    )

                    assert memory_stats["layer"] == true_layer_memory
                    assert (
                        memory_stats["max_memory"]
                        == true_layer_memory + (2 - forward_only) * true_pass_memory
                    )

                    if strict:
                        assert memory_stats["layer"] == num_blocks_layer * BLOCK_SIZE
                        assert (
                            memory_stats["max_memory"]
                            == (num_blocks_layer + (2 - forward_only) * num_blocks_pass)
                            * BLOCK_SIZE
                        )

    # reset memory stats and ensure there is no memory leakage
    assert reset_peak_memory_stats(device).cur_mem == 0
