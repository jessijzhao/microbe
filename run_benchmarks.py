import argparse
import json
import logging
import torch
import pickle
from benchmarks import BenchmarkFactory
from layers import LayerType, LayerFactory
from os.path import exists


logger = logging.getLogger(__name__)

def get_path(layer_name, batch_size, config):
    pickle_name = f"{layer_name}_bs_{batch_size}_runs_{config['num_runs']}_repeats_{config['num_repeats']}"
    return f'./results/raw/{pickle_name}.pkl'

def save_results(layer_name, batch_size, config, runtimes, memory):
    data = {
        'layer_name': layer_name,
        'batch_size': batch_size,
        'config': config,
        'runtime': runtimes,
        'memory': memory
    }
    path = get_path(layer_name, batch_size, config)
    logger.info(f'Saving to: {path}')

    with open(path, 'wb') as handle:
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
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--timer", default="torch", help="use Python, CUDA or PyTorch timer")
    parser.add_argument("--num_runs", default=100, type=int, help="number of benchmarking runs")
    parser.add_argument("--num_repeats", default=20, type=int, help="how many forward/backward passes per run")
    parser.add_argument("--num_warmups", default=10, type=int, help="number of warmups for custom benchmarks")
    parser.add_argument('--cont', action="store_true", help="only run missing experiments")
    parser.add_argument('--batch_sizes', default=[16, 32, 64, 128, 256, 512, 1024, 2048], nargs='+', type=int)
    parser.add_argument('--layers',
        default=[v for k, v in LayerType.__dict__.items() if not k.startswith('__')],
        nargs='+'
    )
    parser.add_argument('--no_save', action="store_true")
    args = parser.parse_args()
    
    # activate logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    conf = {
        "num_warmups": args.num_warmups,
        "num_repeats": args.num_repeats,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize()

    benchmark_run = BenchmarkFactory.create(args.timer)

    for batch_size in args.batch_sizes:
        for layer_set in config["layers"]:
            for layer_name in layer_set["comparison"]:
                temp_conf = {
                    **layer_set["config"],
                    **conf,
                    "num_runs": args.num_runs,
                }

                # skip if we already have the relevant results
                if (
                    layer_name not in args.layers
                    or (args.cont and exists(get_path(layer_name, batch_size, temp_conf)))
                ):
                    if layer_name in args.layers:
                        logger.info(f'Skipping {layer_name} at {batch_size}.')

                else:
                    runtimes = []
                    memory = []
                    for _ in range(args.num_runs):
                        torch.cuda.empty_cache()
                        base_mem = torch.cuda.memory_allocated(device) * 1e-9

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

                        except RuntimeError as e: # OOM
                            runtime  = float('nan')
                            max_memory = float('nan')
                            logger.debug(f'{layer_name} at {batch_size} failed with {e}.')

                        del layer_fun
                        torch.cuda.empty_cache()
                        
                        runtimes.append(runtime)
                        memory.append((base_mem, max_memory))
                        logger.debug(f"Runtime: {runtime:.3f}, base memory: {base_mem:.6f}, max memory: {max_memory:.4f}")

                    if not args.no_save:
                        save_results(
                            layer_name=layer_name,
                            batch_size=batch_size, 
                            config=temp_conf,
                            runtimes=runtimes,
                            memory=memory
                        )

   
if __name__ == "__main__":
    main()
