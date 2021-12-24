"""
    get information of receptive field
"""


def get_receptive_field(neuron_index, layer_info, pad=(0, 0)):
    """
    neuron_index: tuple of length 2 or int represented x axis and y
    layer_info: tuple of length 4 has information of receptive_field
    """
    n, j, rf, start = layer_info
    if isinstance(neuron_index, tuple):
        center_y = start + (neuron_index[1]) * (j)
        center_x = start + (neuron_index[0]) * (j)
    else:
        center_y = start + (neuron_index // n) * (j)
        center_x = start + (neuron_index % n) * (j)
    return (center_x, center_y), (rf / 2, rf / 2)
