import argparse
import json
import torch
import torch.utils.benchmark as benchmark
from layers import LayerFactory, LayerType
from utils import get_layer_set


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_FACTOR = 1e3 #ms
MEM_FACTOR = 1e-9 #GB


def main(args) -> None:

    with open(args.config_file) as config_file:
        config = json.load(config_file)

    layer_set = get_layer_set(args.layer)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        assert(torch.cuda.memory_allocated(device) == 0)

    # setup layer
    layer_fun = LayerFactory.create(
        layer_name=args.layer,
        batch_size=args.batch_size,
        **config[layer_set]
    )
    if not args.forward_only:
        layer_fun.prepare_forward_backward()
        function = layer_fun.forward_backward
    else:
        layer_fun.prepare_forward_only()
        function = layer_fun.forward_only

    if torch.cuda.is_available():
        layer_memory = torch.cuda.max_memory_allocated(device)
    
    # benchmark.Timer performs own warmups
    timer =  benchmark.Timer(
        stmt="function()",
        globals={
            "function": function
        },
        num_threads=1
    )
    runtime = timer.timeit(args.num_repeats).mean

    # get max memory allocated
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated(device)
    else:
        max_memory = float('nan')
    
    print(runtime * TIME_FACTOR, layer_memory * MEM_FACTOR, max_memory * MEM_FACTOR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('layer',
        choices=[v for k, v in LayerType.__dict__.items() if not k.startswith('__')],
    )
    parser.add_argument('--batch_size', type=int)
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="config.json"
    )
    parser.add_argument(
        "--num_repeats", 
        default=20, 
        type=int, 
        help="how many forward/backward passes per run"
    )
    parser.add_argument(
        "--forward_only",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)