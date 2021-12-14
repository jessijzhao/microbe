def get_layer_set(layer):
    """Layers in the same layer set share a config"""
    layer_set = layer.replace("gsm_dp", "").replace("gsm_", "").replace("dp", "")
    if layer_set in ['rnn', 'gru', 'lstm']:
        layer_set = 'rnn_base'
    return layer_set
