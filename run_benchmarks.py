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


def main() -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('config.json') as config_file:
        config = json.load(config_file)

    torch.cuda.synchronize()

    conf = {
        "num_warmups": config["num_warmups"],
        "num_runs": config["num_runs"],
    }

    for benchmark_type in config["benchmarks"]:
        benchmark_run = BenchmarkFactory.create(benchmark_type)

        for layer_set in config["layers"]:
            print(json.dumps(layer_set["config"], indent=4))

            for comparison in layer_set["comparison"]:
                assert(len(comparison) == 2)
                print(f"Comparing {comparison} ...\n")

                results = pd.DataFrame(columns=["Forward only", "Forward backward"])
                std = copy.deepcopy(results)

                for layer_type in comparison:
                    runtimes = []
                    for _ in range(config["num_repeats"]):
                        
                        # setup layer
                        layer_fun = LayerFactory.create(
                            layer_type=layer_type,
                            **layer_set["config"]
                        )
                        layer_fun.layer.to(device)
                        
                        # benchmark forward_only
                        with torch.no_grad():
                            layer_fun.prepare_forward_only()
                            forward_only_runtime = benchmark_run.run(
                                function=layer_fun.forward_only_no_hooks,
                                **conf, 
                            )
                        logger.info(f"Runtime for {layer_type} on forward_only: {forward_only_runtime} ms")
                        
                    
                        # benchmark forward_backward
                        layer_fun.prepare_forward_backward()
                        forward_backward_runtime = benchmark_run.run(
                            function=layer_fun.forward_backward,
                            **conf, 
                        )
                        logger.info(f"Runtime for {layer_type} on forward_backward: {forward_backward_runtime} ms")
                        
                        del layer_fun
                        runtimes.append((forward_only_runtime, forward_backward_runtime))

                    runtimes = np.array(runtimes)
                    forward_only_mean, forward_backward_mean = np.mean(runtimes, axis=0)
                    forward_only_std, forward_backward_std = np.std(runtimes, axis=0)
                     
                    results = results.append(
                        pd.Series(
                            [forward_only_mean, forward_backward_mean],
                            name=layer_type,
                            index=results.columns
                        )
                    )
                 
                    std = std.append(
                        pd.Series(
                            [forward_only_std, forward_backward_std],
                            name=layer_type,
                            index=results.columns
                        )
                    )
                 
                results["Factor"] = results["Forward only"] / results["Forward backward"]
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
                print(f"Mean runtime across {config['num_repeats']} runs:")
                print(results, "\n")

                print(f"Standard deviation across {config['num_repeats']} runs:")
                print(std, "\n")
                
   
if __name__ == "__main__":
    main()
