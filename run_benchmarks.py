import argparse
import json
import logging
import subprocess
from os.path import exists
from typing import Any, Dict

from layers import LayerType
from utils import get_layer_set, get_path, save_results


logger = logging.getLogger(__name__)


def run_benchmark(layer: LayerType, batch_size: int, args) -> Dict[str, Any]:
    results = []

    for _ in range(args.num_runs):
        cmd = f"CUDA_VISIBLE_DEVICES=0 python3 -W ignore benchmark_layer.py {layer} --batch_size {batch_size} -c {args.config_file} --num_repeats {args.num_repeats}"
        if args.forward_only:
            cmd += " --forward_only"

        logger.info(f"Starting {cmd}")
        out = subprocess.run(
            [cmd],
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        if out.returncode == 0:
            runtime, layer_mem, max_memory = [
                float(num) for num in out.stdout.split(" ")
            ]
        else:
            # OOM error
            logger.debug(out.stderr)
            runtime, layer_mem, max_memory = [float("nan") for _ in range(3)]

        res = {"runtime": runtime, "layer_memory": layer_mem, "max_memory": max_memory}
        results.append(res)
        logger.info(res)

    return results


def main(args) -> None:
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # get config for each layer
    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    for layer in args.layers:
        for batch_size in args.batch_sizes:

            # skip benchmark if applicable
            if args.cont and exists(
                get_path(
                    layer=layer,
                    batch_size=batch_size,
                    num_runs=args.num_runs,
                    num_repeats=args.num_repeats,
                )
            ):
                logger.info(f"Skipping {layer} at {batch_size} - already exists.")
                continue

            # run the benchmark
            results = run_benchmark(layer=layer, batch_size=batch_size, args=args)

            # save the benchmark results
            if not args.no_save:
                save_results(
                    layer=layer,
                    batch_size=batch_size,
                    num_runs=args.num_runs,
                    num_repeats=args.num_repeats,
                    results=results,
                    config=config[get_layer_set(layer)],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.json")
    parser.add_argument(
        "--layers",
        choices=[v for k, v in LayerType.__dict__.items() if not k.startswith("__")],
        default="all",
        nargs="+",
    )
    parser.add_argument(
        "--batch_sizes",
        default=[16, 32, 64, 128, 256, 512, 1024, 2048],
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--num_runs", default=100, type=int, help="number of benchmarking runs"
    )
    parser.add_argument(
        "--num_repeats",
        default=20,
        type=int,
        help="how many forward/backward passes per run",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--forward_only", action="store_true", help="only run forward passes"
    )
    parser.add_argument("--random_seed", type=int)
    parser.add_argument(
        "--cont", action="store_true", help="only run missing experiments"
    )
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    main(args)
