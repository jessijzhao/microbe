import argparse
import json
from typing import Callable, Tuple

import torch
import torch.utils.benchmark as benchmark

from layers import LayerFactory, LayerType
from utils import Memory, get_layer_set, reset_peak_memory_stats


TIME_FACTOR = 1e3  # ms (milliseconds)
MEM_FACTOR = 1e-9  # GB (gigabytes)


def run_layer_benchmark(
    layer_name: LayerType,
    batch_size: int,
    num_repeats: int,
    forward_only: bool,
    create_layer: Callable = LayerFactory.create,
    **kwargs,
) -> Tuple[float, Memory]:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        assert reset_peak_memory_stats(device).cur_mem == 0

    # setup layer
    layer_fun = create_layer(
        layer_name=layer_name,
        batch_size=batch_size,
        **kwargs,
    )

    if forward_only:
        benchmark_fun = layer_fun.forward_only
    else:
        benchmark_fun = layer_fun.forward_backward

    # get memory allocated and reset memory statistics
    memory_stats = layer_fun.to(device=device, forward_only=forward_only)

    # benchmark.Timer performs its own warmups
    timer = benchmark.Timer(
        stmt="benchmark_fun()", globals={"benchmark_fun": benchmark_fun}, num_threads=1
    )
    runtime = timer.timeit(num_repeats).mean

    # get max memory allocated and reset memory statistics
    memory_stats["max_memory"] = reset_peak_memory_stats(device).prev_max_mem

    return runtime, memory_stats


def main(args) -> None:

    with open(args.config_file) as config_file:
        config = json.load(config_file)

    runtime, memory_stats = run_layer_benchmark(
        layer_name=args.layer,
        batch_size=args.batch_size,
        num_repeats=args.num_repeats,
        forward_only=args.forward_only,
        **config[get_layer_set(args.layer)],
    )
    print(
        runtime * TIME_FACTOR,
        memory_stats["layer"] * MEM_FACTOR,
        memory_stats["max_memory"] * MEM_FACTOR,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.json")
    parser.add_argument(
        "layer",
        choices=[v for k, v in LayerType.__dict__.items() if not k.startswith("__")],
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--num_repeats",
        default=20,
        type=int,
        help="how many forward/backward passes to run",
    )
    parser.add_argument(
        "--forward_only", action="store_true", help="only run forward passes"
    )
    args = parser.parse_args()
    main(args)
