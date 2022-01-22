import math
import time
import torch
import unittest

from benchmark_layer import run_layer_benchmark
from layers import LayerFactory
from utils import device, reset_peak_memory_stats

NUM_REPEATS = [10, 20, 50]

class BenchmarkLayerTest(unittest.TestCase):
    
    def test_runtime_benchmark(self):
        for duration in [0, 0.005, 0.01, 0.1]:
            for num_repeats in NUM_REPEATS:
                runtime, layer_memory, max_memory = run_layer_benchmark(
                    lambda: time.sleep(duration), num_repeats
                )
                # account for small variations
                assert abs(runtime - duration) < 0.001

                # check that no memory allocation took place
                if torch.cuda.is_available():
                    assert(layer_memory == 0 and max_memory == 0)

    def test_memory_benchmark(self):
        """CUDA memory is allocated in blocks, where block sizes vary across 
        kernels. New CUDA memory is allocated for each new tensor."""

        if torch.cuda.is_available():
            # reset memory stats, ensure no memory leakage
            assert(reset_peak_memory_stats(device)[1] == 0)

            # find the block size by creating a tensor of size 1 byte
            tiny_tensor = torch.zeros(1, dtype=torch.int8, device=device)
            BLOCK_SIZE = torch.cuda.max_memory_allocated(device)
            del tiny_tensor
            
            # both base-2 and base-10 numbers of bytes
            for num_bytes in [64, 128, 256, 512, 1024, 2048] + [100, 500, 1000, 2000]:
                # whether to have a layer or not
                for layer_size in [0, 1]:
                    for num_repeats in NUM_REPEATS:

                        # create a (potentially empty) dummy layer
                        layer = torch.zeros(layer_size, dtype=torch.int8, device=device)

                        # allocate memory (int8 = 1 byte)
                        runtime, layer_memory, max_memory = run_layer_benchmark(
                            lambda: torch.zeros(num_bytes, dtype=torch.int8, device=device),
                            num_repeats
                        )
                        num_blocks_layer = math.ceil(layer_size / BLOCK_SIZE) 
                        assert(layer_memory == num_blocks_layer * BLOCK_SIZE)
                        assert (
                            max_memory
                            == (num_blocks_layer + math.ceil(num_bytes / BLOCK_SIZE)) * BLOCK_SIZE
                        )


                        del layer
            
                        # reset memory stats and ensure there is no memory leakage
                        assert(reset_peak_memory_stats(device)[1] == 0)