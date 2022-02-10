import math
import time
from typing import List

import pytest
import torch
from helpers import get_actual_memory_allocated, get_n_byte_tensor

from benchmark_layer import run_layer_benchmark
from layers import LayerFactory
from utils import device, reset_peak_memory_stats


# number of repeats to test runtime and memory benchmarks for
NUM_REPEATS = [10, 20, 50]


class FakeLayer(LayerFactory.Layer):
    """Fake layer to test runtime and memory benchmarking

    Kwargs:
        runtime: duration for one forward or backward pass
        pass_memory: memory used during a forward or backward pass
        layer_memory: memory taken up by the inputs and layer
    """

    def __init__(self, **kwargs) -> None:
        self.runtime = kwargs.get("runtime", 0)
        self.pass_memory = kwargs.get("pass_memory", 0)
        self.layer = get_n_byte_tensor(kwargs.get("layer_memory", 0))

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def forward_only(self) -> torch.Tensor:
        """Wait for self.duration and allocate self.max_memory bytes"""
        time.sleep(self.runtime)
        tensor = get_n_byte_tensor(self.pass_memory)
        return tensor

    def forward_backward(self) -> None:
        """2x runtime and memory of forward_only"""
        _ = self.forward_only()
        _ = self.forward_only()


@pytest.mark.parametrize(
    "duration_list",
    [([0, 0.005, 0.01, 0.05])],
)
def test_runtime_benchmark(duration_list: List[float]) -> None:
    """Test runtime benchmarks on a dummy layer.

    Args:
        - duration_list: list of durations (s) to test runtime benchmarks for
    """
    for duration in duration_list:
        for num_repeats in NUM_REPEATS:
            for forward_only in [0, 1]:
                runtime, layer_memory, max_memory = run_layer_benchmark(
                    layer_name="",
                    batch_size=0,
                    num_repeats=num_repeats,
                    forward_only=forward_only,
                    create_layer=FakeLayer,
                    runtime=duration,
                )
                # account for small variations
                assert abs(runtime - ((2 - forward_only) * duration)) < 0.001

                # check that no memory allocation took place
                if torch.cuda.is_available():
                    assert layer_memory == 0 and max_memory == 0


@pytest.mark.parametrize(
    "pass_bytes_list, layer_bytes_list",
    [([1, 128, 256, 500, 512, 1024, 2000], [0, 128, 500])],
)
def test_memory_benchmark(
    pass_bytes_list: List[int], layer_bytes_list: List[int], strict: bool = True
) -> None:
    """Test CUDA max_memory benchmarks on a dummy layer.

    Notes: During the experiments included in the paper, CUDA memory is allocated
    in blocks, where block sizes vary across kernels. New CUDA memory is
    allocated for each new tensor. The strict test will fail under a different
    allocation scheme.

    Args:
        - pass_bytes_list: list of bytes that each forward or backward pass allocates
        - layer_bytes_list: list of bytes that the input and layer allocate
        - strict: whether to predict each measurement based on block size
    """

    if torch.cuda.is_available():

        if strict:
            # find the block size by creating a tensor of size 1 byte
            tiny_tensor = get_n_byte_tensor(1)
            BLOCK_SIZE = torch.cuda.max_memory_allocated(device)
            del tiny_tensor

        # bytes allocated during forward/backward pass
        for pass_bytes in pass_bytes_list:
            true_pass_memory = get_actual_memory_allocated(pass_bytes)
            if strict:
                num_blocks_pass = math.ceil(pass_bytes / BLOCK_SIZE)

            # bytes allocated for the layer, inputs, etc.
            for layer_bytes in layer_bytes_list:
                true_layer_memory = get_actual_memory_allocated(layer_bytes)
                if strict:
                    num_blocks_layer = math.ceil(layer_bytes / BLOCK_SIZE)

                for num_repeats in NUM_REPEATS:
                    for forward_only in [0, 1]:
                        # reset memory stats and ensure there is no memory leakage
                        assert reset_peak_memory_stats(device)[1] == 0

                        runtime, layer_memory, max_memory = run_layer_benchmark(
                            layer_name="",
                            batch_size=0,
                            num_repeats=num_repeats,
                            forward_only=forward_only,
                            create_layer=FakeLayer,
                            layer_memory=layer_bytes,
                            pass_memory=pass_bytes,
                        )
                    assert layer_memory == true_layer_memory
                    assert (
                        max_memory
                        == true_layer_memory + (2 - forward_only) * true_pass_memory
                    )

                    if strict:
                        assert layer_memory == num_blocks_layer * BLOCK_SIZE
                        assert (
                            max_memory
                            == (num_blocks_layer + (2 - forward_only) * num_blocks_pass)
                            * BLOCK_SIZE
                        )

        # reset memory stats and ensure there is no memory leakage
        assert reset_peak_memory_stats(device)[1] == 0
