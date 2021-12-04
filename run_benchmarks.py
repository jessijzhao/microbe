import argparse
import logging
import copy
import torch
from layers import LayerFactory
from benchmarks import BenchmarkFactory
import json
import torch.nn as nn
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def append_times(runtimes, df, layer_type, fun=np.mean):
    runtimes = np.array(runtimes)
    return df.append(
        pd.Series(
            fun(runtimes, axis=0),
            name=layer_type,
            index=df.columns
        )
    ) 


def generate_df(results, comparison):
    results["Factor"] = results["Forward backward"] / results["Forward only"]
    results = results.append(
        pd.Series(
            (
                results.loc[[comparison[1]]].values/
                results.loc[[comparison[0]]].values
            )[0],
            name="Factor",
            index=results.columns
        )
    )
    return results


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
    args = parser.parse_args()
    
    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    conf = {
        "num_warmups": config["num_warmups"],
        "num_runs": config["num_runs"],
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize()

    benchmark_run = BenchmarkFactory.create(config["benchmark"])

    for layer_set in config["layers"]:
        print(json.dumps(layer_set["config"], indent=4))

        for comparison in layer_set["comparison"]:
            assert(len(comparison) == 2)
            print(f"Comparing {comparison} ...\n")

            for batch_size in config["batch_sizes"]:

                r_median = pd.DataFrame(columns=["Forward only", "Forward backward"])
                m_median = copy.deepcopy(r_median)

                for layer_type in comparison:
                    runtimes = []
                    memory = []
                    for _ in range(config["num_repeats"]):

                        # setup layer
                        layer_fun = LayerFactory.create(
                            layer_type=layer_type,
                            batch_size=batch_size,
                            **layer_set["config"]
                        )

                        # benchmark forward_only
                        with torch.no_grad():
                            layer_fun.prepare_forward_only()
                            forward_only_runtime, forward_only_memory= benchmark_run.run(
                                function=layer_fun.forward_only_no_hooks,
                                **conf, 
                            )
                        logger.info(f"Runtime for {layer_type} on forward_only: {forward_only_runtime} ms")
                        del layer_fun
                        torch.cuda.empty_cache() 

                       # setup layer
                        layer_fun = LayerFactory.create(
                            layer_type=layer_type,
                            batch_size=batch_size,
                            **layer_set["config"]
                        )
                       
                        # benchmark forward_backward
                        layer_fun.prepare_forward_backward()
                        forward_backward_runtime, forward_backward_memory = benchmark_run.run(
                            function=layer_fun.forward_backward,
                            **conf, 
                        )
                        logger.info(f"Runtime for {layer_type} on forward_backward: {forward_backward_runtime} ms")
                        
                        del layer_fun
                        torch.cuda.empty_cache() 
                        runtimes.append((forward_only_runtime, forward_backward_runtime))
                        memory.append((forward_only_memory, forward_backward_memory))

                    m_median = append_times(memory, m_median, layer_type, fun=np.median)
                    r_median = append_times(runtimes, r_median, layer_type, fun=np.median)

                m_median = generate_df(m_median, comparison)
                r_median = generate_df(r_median, comparison)
                print(f"Median max memory (GB) across {config['num_repeats']} runs for batch_size {batch_size}:")
                print(m_median, "\n")

                print(f"Median runtime (ms) across {config['num_repeats']} runs for batch_size {batch_size}:")
                print(r_median, "\n")

   
if __name__ == "__main__":
    main()
