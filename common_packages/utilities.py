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
        raise TypeError(
            "Invalid regularizer type. Valid types are 'None', 'L1', or 'L2'"
        )


def get_num_parameters_per_layer(num_qumodes):
    """Find the number of parameters in a layer for a given number of qumodes"""
    displacement_parameters = 2 * num_qumodes
    squeezing_parameters = 2 * num_qumodes
    kerr_parameters = num_qumodes
    interferometer_parameters = 2 * (
        num_qumodes + 2 * num_qumodes * (num_qumodes - 1) / 2
    )
    return (
        displacement_parameters
        + squeezing_parameters
        + kerr_parameters
        + interferometer_parameters
    )
