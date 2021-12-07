# microbe

A set of **microbe**nchmarks that compares [Opacus](https://github.com/pytorch/opacus) layers (both [basic modules](https://github.com/pytorch/opacus/tree/main/opacus/grad_sample) and [more complex layers](https://github.com/pytorch/opacus/tree/main/opacus/layers)) to their respective [torch.nn](https://pytorch.org/docs/stable/nn.html) counterparts.

## Contents

- [run_benchmarks.py](run_benchmarks.py) runs all benchmarks in a given config file and writes out the results as pickle files to results/raw

- [config.json](config.json) contains an example JSON config to run benchmarks with

- [layers.py](layers.py) implements the generation of each layer and its input as well as the forward/backward pass for each layer

- [benchmarks.py](benchmarks.py) provides benchmarks using Python's timeit, CUDA, and PyTorch's timer respectively (default: PyTorch's benchmarking itmer)

- [results/analysis.ipynb](results/analysis.ipynb) analyzes and plots the benchmark results


## Benchmarks

For each layer and batch size in [config.json](config.json), [run_benchmarks.py](run_benchmarks.py) will do the following:
```
Do this num_runs times:
    Init layer, one batch of random input, one batch of random "labels"

    Reset memory statistics
    Start timer

    Do this num_repeats times:
        preds = layer(input)
        loss = self.cross_entropy(preds, self.labels)
        loss.backward()

    Stop timer

    Return elapsed time / num_repeats and maximum allocated memory
```

## Layers

All layers including Opacus layers follow `torch.nn`'s interface with the same default values if not specified in [config.json](config.json).

A note on `input_shape` in [config.json](config.json): parameters that are shared between the model and the input are listed separately. Therefore, the actual input shape will vary:

- Linear: `(batch_size, *input_shape, in_features)`

- Convolutional: `(batch_size, in_channels, *input_shape)`

- LayerNorm:
    - Input: `(batch_size, *input_shape)`
    - Normalized shape: `(input_shape[-D:])`

- InstanceNorm: `(batch_size, num_features, *input_shape)`

- GroupNorm: `(batch_size, num_channels, *input_shape)`

- Embedding: `(batch_size, *input_shape)`

- MultiheadAttention:
    - If not batch_first: `(targ_seq_len, batch_size, embed_dim)`
    - Else: `(batch_size, targ_seq_len, embed_dim)`

- RNN, GRU, LSTM:
    - If not batch_first: `(seq_len, batch_size, input_size)`
    - Else: `(batch_size, seq_len, input_size)`


## Usage

```
usage: run_benchmarks.py [-h] [-c CONFIG_FILE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
  -v, --verbose
 ```