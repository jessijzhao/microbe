import argparse
import logging
import copy
import torch
import pickle
from layers import LayerFactory
from benchmarks import BenchmarkFactory
import json
import torch.nn as nn
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def save_results(layer_name, batch_size, config, runtimes, memory):
    data = {
        'layer_name': layer_name,
        'batch_size': batch_size,
        'config': config,
        'runtime': runtimes,
        'memory': memory
    }
    pickle_name = f"{layer_name}_bs_{batch_size}_runs_{config['num_repeats']}_repeats_{config['num_runs']}"
    full_path = './results/raw/' + pickle_name + '.pkl'
    print('Saving to: ', full_path)

    with open(full_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:

    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        type=str,
        default="config.json"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true"
    )
    args = parser.parse_args()
    
    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    conf = {
        "num_warmups": config["num_warmups"],
        "num_repeats": config["num_repeats"],
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize()

    benchmark_run = BenchmarkFactory.create(config["benchmark"])

    for layer_set in config["layers"]:
        if args.verbose:
            print(json.dumps(layer_set["config"], indent=4))

        for batch_size in config["batch_sizes"]:
            for layer_name in layer_set["comparison"]:
                runtimes = []
                memory = []
                for _ in range(config["num_runs"]):

                    # setup layer
                    layer_fun = LayerFactory.create(
                        layer_name=layer_name,
                        batch_size=batch_size,
                        **layer_set["config"]
                    )
                   
                    try:
                        # benchmark forward_backward
                        layer_fun.prepare_forward_backward()
                        runtime, max_memory = benchmark_run.run(
                            function=layer_fun.forward_backward,
                            **conf
                        )
                        del layer_fun
                        torch.cuda.empty_cache()

                    except RuntimeError: # OOM
                        runtime  = float('nan')
                        max_memory = float('nan')

                    runtimes.append(runtime)
                    memory.append(max_memory)

                save_results(
                    layer_name=layer_name,
                    batch_size=batch_size, 
                    config={
                        **layer_set["config"],
                        **conf,
                        "num_runs": config['num_runs']
                    },
                    runtimes=runtimes,
                    memory=memory
                )

   
if __name__ == "__main__":
    main()
