"""CV Hybrid Networks 2022"""
import tensorflow as tf
from deap_tf.mappers import PWBMapper
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class PWBLinearLayer(keras.layers.Layer):
    """
    This class implements a photonic weight bank simulated linear dense layer for Tensorflow.
    Weights and biases should be limited to [-1,1] to maintain physical accuracy."""

    def __init__(self, num_outputs, precision=127, activation=None, name="PWBLinearLayer"):
        """Initialize PWB Linear Layer

        Args:
            num_outputs (int): Number of outputs for the layer
            precision (int, optional): PWB precision. Defaults to 127.
            activation (str, optional): Activation function name. Defaults to None.
            name (str, optional): Name of layer
        """
        super(PWBLinearLayer, self).__init__()
        self.num_outputs = num_outputs
        self.pwb_mapper = PWBMapper
        self.precision = precision
        self.activation = keras.activations.get(activation)
        self.constraint = lambda t: tf.clip_by_value(t, -1.0, 1.0)

    def build(self, input_shape):
        """Build the weights and biases for the PWB layer

        Overrides keras.layers.Layer class's build method

        Args:
            input_shape (list): List describing the input shape to the layer
        """
        w_init = tf.random_normal_initializer()
        self.num_inputs = input_shape[-1]
        self.weight = self.add_weight(
            "kernel",
            shape=[self.num_inputs, self.num_outputs],
            initializer=w_init,
            dtype=tf.float32,
            trainable=False,
            constraint=self.constraint,
        )
        self.bias = self.add_weight(
            "bias",
            shape=[self.num_outputs],
            initializer=w_init,
            dtype=tf.float32,
            trainable=False,
            constraint=self.constraint,
        )
        self.pwb_mapper.setPrecision(self.precision)
        self.neurons = [self.pwb_mapper.build(weight) for weight in tf.transpose(self.weight)]
        self.bias_neuron = self.pwb_mapper.build(self.bias)

    def run_pwb(self, inputs):
        """Simulate the PWB

        Args:
            inputs (?): Inputs to the PWB

        Returns:
            ?: Outputs from the PWB
        """
        res = tf.convert_to_tensor([neuron.step(inputs) for neuron in self.neurons])
        b_inputs = tf.eye(self.bias.shape[0], dtype=tf.float32)
        bias = [tf.convert_to_tensor(self.bias_neuron.step(b_input)) for b_input in b_inputs]
        return res + bias

    def call(self, inputs):
        """Call the layer

        Overrides keras.layers.Layer class's call method
        First we have to update weights in our photonic neurons

        Args:
            inputs (np.array??): Input matrix

        Returns:
            np.array??: Output matrix
        """
        for neuron, weight in zip(self.neurons, tf.transpose(self.weight)):
            self.pwb_mapper.updateWeights(neuron, weight)
        self.pwb_mapper.updateWeights(self.bias_neuron, self.bias)
        if len(inputs.get_shape()) > 1:
            pwb_output = []
            for data in inputs:
                pwb_output.append(self.run_pwb(data))
        else:
            pwb_output = self.run_pwb(inputs)
        return self.activation(tf.stack(pwb_output))
