# microbe

A set of **microbe**nchmarks that compares [Opacus](https://github.com/pytorch/opacus) layers (both [basic modules](https://github.com/pytorch/opacus/tree/main/opacus/grad_sample) and [more complex layers](https://github.com/pytorch/opacus/tree/main/opacus/layers)) to their respective [torch.nn](https://pytorch.org/docs/stable/nn.html) counterparts.

## Contents

- [run_benchmarks.py](run_benchmarks.py) runs all benchmarks in a given config file and writes out the results as pickle files to results/raw

- [config.json](config.json) contains an example JSON config to run benchmarks with

- [layers.py](layers.py) implements each layer, its input, and the forward/backward pass for each layer

- [results](results) contains notebooks for analyzing and plotting the benchmarking results


## Benchmarks

For each layer and batch size in [config.json](config.json), [run_benchmarks.py](run_benchmarks.py) will do the following:
```
Do this num_runs times:
    Init layer, one batch of random input, one batch of random "labels"

    Reset memory statistics
    Start timer

    Do this num_repeats times:
        preds = layer(input)
        loss = self.criterion(preds, self.labels)
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

If saving results, ensure that `results/raw` directory exists.

```
usage: run_benchmarks.py [-h] [-c CONFIG_FILE] [-v] [--num_runs NUM_RUNS]
                         [--num_repeats NUM_REPEATS] [--cont]
                         [--batch_sizes BATCH_SIZES [BATCH_SIZES ...]]
                         [--layers {linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} [{linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} ...]]
                         [--forward_only] [--no_save]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
  -v, --verbose
  --num_runs NUM_RUNS   number of benchmarking runs
  --num_repeats NUM_REPEATS
                        how many forward/backward passes per run
  --cont                only run missing experiments
  --batch_sizes BATCH_SIZES [BATCH_SIZES ...]
  --layers {linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} [{linear,gsm_linear,conv,gsm_conv,layernorm,gsm_layernorm,instancenorm,gsm_instancenorm,groupnorm,gsm_groupnorm,embedding,gsm_embedding,mha,dpmha,gsm_dpmha,rnn,dprnn,gsm_dprnn,gru,dpgru,gsm_dpgru,lstm,dplstm,gsm_dplstm} ...]
  --forward_only
  --no_save
```


## Tests

```python -m pytest tests/```