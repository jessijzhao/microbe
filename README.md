# microbe

A set of **microbe**nchmarks that compares [Opacus](https://github.com/pytorch/opacus) layers (both [basic modules](https://github.com/pytorch/opacus/tree/main/opacus/grad_sample) and [more complex layers](https://github.com/pytorch/opacus/tree/main/opacus/layers)) to their respective [torch.nn](https://pytorch.org/docs/stable/nn.html) counterparts.

## Contents

- [run_benchmarks.py](run_benchmarks.py) runs all benchmarks in a given config file

- [config.json](config.json) contains an example JSON config to run benchmarks with

- [layers.py](layers.py) implements the forward/backward pass for each layer

- [benchmarks.py](benchmarks.py) provides benchmarks using Python's timeit, CUDA, and pytorch's timer respectively


## Usage

```
usage: run_benchmarks.py [-h] [-c CONFIG_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
 ```