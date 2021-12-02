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

                results = pd.DataFrame(columns=["Inference", "Training"])

                for model_type in comparison:
                    # setup model
                    model_fun = ModelFactory.create(
                        model_type=model_type,
                        **model_set["config"]
                    )
                    model_fun.model.to(device)
                    
                    # benchmark inference
                    model_fun.prepare_inference()
                    inf_runtime = benchmark_run.run(
                        function=model_fun.inference,
                        **conf, 
                    )
                    logger.info(f"Runtime for {model_type} on inference: {inf_runtime} ms")
                    
                
                    # benchmark training
                    model_fun.prepare_training()
                    train_runtime = benchmark_run.run(
                        function=model_fun.training,
                        **conf, 
                    )
                    logger.info(f"Runtime for {model_type} on training: {train_runtime} ms")
                    
                    del model_fun

                    results = results.append(
                        pd.Series(
                            [inf_runtime, train_runtime],
                            name=model_type,
                            index=results.columns
                        )
                    )
                 
                results["Factor"] = results["Training"] / results["Inference"]
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
