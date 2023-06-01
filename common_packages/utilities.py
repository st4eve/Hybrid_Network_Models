import math

from keras import regularizers


def get_regularizer(regularizer_string):
    """Gets the regularizer from the string"""
    if regularizer_string is None:
        return None
    regularizer_type = regularizer_string.split("=")[0]
    value = float(regularizer_string.split("=")[1])
    if regularizer_type == "L1":
        return regularizers.L1(l1=value)
    elif regularizer_type == "L2":
        return regularizers.L2(l2=value)
    else:
        raise TypeError("Invalid regularizer type. Valid types are 'None', 'L1', or 'L2'")


def get_num_parameters_per_quantum_layer(num_qumodes):
    """Find the number of parameters in a layer for a given number of qumodes"""
    displacement_parameters = 2 * num_qumodes
    squeezing_parameters = 2 * num_qumodes
    kerr_parameters = num_qumodes
    interferometer_parameters = 2 * (num_qumodes + 2 * num_qumodes * (num_qumodes - 1) / 2)
    return displacement_parameters + squeezing_parameters + kerr_parameters + interferometer_parameters


def get_equivalent_classical_layer_size(num_qumodes, num_input_neurons, num_output_neurons):
    """Get the number of classical nodes for an equivalent quantum circuit

    Args:
        num_qumodes (int): Number of qumodes
        num_input_neurons (int): Number of input neurons to the classical/quantum layer, typically num_qumodes, or 2*num_qumodes
        num_output_neurons (int): Number of neurons in the layer after the classical/quantum layer

    Returns:
        int: Number of neurons for equivalent classical layer
    """
    num_qumode_params = get_num_parameters_per_quantum_layer(num_qumodes) + num_qumodes * num_output_neurons
    num_params_per_classical_neuron = num_input_neurons + num_output_neurons + 1
    return math.ceil(num_qumode_params / num_params_per_classical_neuron)



