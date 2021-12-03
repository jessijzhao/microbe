import logging
import torch
from models import ModelFactory
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

        for model_set in config["models"]:
            print(json.dumps(model_set["config"], indent=4))

            for comparison in model_set["comparison"]:
                assert(len(comparison) == 2)
                print(f"Comparing {comparison} ...")

                results = pd.DataFrame(columns=["Forward only", "Forward backward"])

                for model_type in comparison:
                    # setup model
                    model_fun = ModelFactory.create(
                        model_type=model_type,
                        **model_set["config"]
                    )
                    model_fun.model.to(device)
                    
                    # benchmark forward_only
                    with torch.no_grad():
                        model_fun.prepare_forward_only()
                        forward_only_runtime = benchmark_run.run(
                            function=model_fun.forward_only_no_hooks,
                            **conf, 
                        )
                    logger.info(f"Runtime for {model_type} on forward_only: {forward_only_runtime} ms")
                    
                
                    # benchmark forward_backward
                    model_fun.prepare_forward_backward()
                    forward_backward_runtime = benchmark_run.run(
                        function=model_fun.forward_backward,
                        **conf, 
                    )
                    logger.info(f"Runtime for {model_type} on forward_backward: {forward_backward_runtime} ms")
                    
                    del model_fun

                    results = results.append(
                        pd.Series(
                            [forward_only_runtime, forward_backward_runtime],
                            name=model_type,
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
                print(results, "\n")
                
   
if __name__ == "__main__":
    main()
