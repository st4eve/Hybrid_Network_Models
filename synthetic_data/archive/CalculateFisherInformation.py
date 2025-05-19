# Calculate Fisher Information Matrix for a given model
#%% Imports
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import keras
import sys
sys.path.append("../")
from synthetic_data.archive.quantum_base import Net
import copy
from common_packages.utilities import get_equivalent_classical_layer_size, get_num_parameters_per_quantum_layer
tf.compat.v1.enable_eager_execution()
from numba import jit, njit
import scipy
import pandas as pd
from common_packages.CV_quantum_layers import QuantumLayer
from tqdm import tqdm


#%%
class Calculate_Fisher_Information_Matrix():
    def __init__(
        self,
        model_func,
        x_train,
        n_iter = 10
    ):
        """Helper class to calculate the fisher information matrix for a given model

        Args:
            model_func (Keras Model Initialization Function): Function to generate model to be tested
            x_train (arr): Input data for the provided model
            n_iter (int, optional): Number of random model initializations to calculate the fisher information over. Defaults to 10.
        """
        self.model_func = model_func
        self.model = self.model_func()
        self.test_model = self.model_func()
        self.x_train = np.array(x_train)
        self.test_model(self.x_train[0:1])
        self.n_iter = n_iter
        self.inputs = np.random.uniform(0, 1, size=[n_iter] + list(self.x_train.shape))
        self.n_classes = self.test_model.layers[-1].output_shape[-1]
        
        self.num_params = self.calc_total_params(self.test_model.trainable_variables)
        self.param_vol = self.calc_param_vol(self.test_model.layers)
        self.grads = self.calc_all_grads()
        self.fishers = self.calc_fisher()
        self.fhat, self.fisher_trace = self.process_fishers()
      
      
    @tf.function
    def calc_jac_classical(self, inputs, model):
        with tf.GradientTape() as t:
            t.watch(model.trainable_variables)
            outputs = model(inputs)
        return t.jacobian(outputs, model.trainable_variables)
 
 
    def calc_jac_quantum(self, inputs, model):
        with tf.GradientTape() as t:
            t.watch(model.trainable_variables)
            outputs = model(inputs)
        return t.jacobian(outputs, model.trainable_variables)
            
        
    def calc_jac(self, inputs, model):
        if True in ['quantum' in l.name for l in model.layers]:
            return self.calc_jac_quantum(inputs, model)
        else:
            return self.calc_jac_classical(inputs, model)


    def calc_total_params(self, trainable_variables):
        length = sum([np.prod(v.shape) for v in trainable_variables])
        return length

    def calc_param_vol(self, layers):
        param_vol = 1
        for l in layers:
            if 'quantum_layer_multi_qunode' in l.name:
                param_vol *= l.param_vol
            elif 'sequential' in l.name:
                param_vol *= self.calc_param_vol(l.layers)
            elif 'dense' in l.name:
                if type(l.kernel_initializer) is keras.initializers.initializers_v2.GlorotUniform:
                    param_vol *= np.sqrt(6 / (l.input_shape[-1] + l.output_shape[-1]))**(l.input_shape[-1] * l.output_shape[-1])
                else:
                    param_vol *= 1
        return param_vol
                
    def reinitialize_model(self, model):
        for l in model.layers:
            if 'quantum_layer_multi_qunode' in l.name:
                l.initialize_circuit()
            elif 'sequential' in l.name:
                self.reinitialize_model(l)
            else:
                if hasattr(l,"kernel_initializer"):
                    l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
                if hasattr(l,"bias_initializer") and l.use_bias:
                    l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
                if hasattr(l,"recurrent_initializer"):
                    l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))

    def calc_fisher(self):
        grads = self.grads
        total_params = self.num_params
        n_classes = self.n_classes
        fishers = np.zeros([len(grads), total_params, total_params])
        for i in tqdm(range(len(grads)), desc='Calculating Fishers'):
            grad = grads[i]
            temp_sum = np.zeros([n_classes, total_params, total_params])
            for j in range(n_classes):
                temp_sum[j] += np.outer(grad[j], np.transpose(grad[j]))
            fishers[i] = np.sum(temp_sum, axis=0)
        return fishers


    def process_fishers(self):
        fishers = self.fishers
        total_params = self.num_params
        param_vol = self.param_vol
        n_iter = self.n_iter
        num_datapoints = len(self.x_train)
        
        fisher_trace = np.trace(np.mean(fishers, axis=0))
        fishers = np.average(np.reshape(fishers, [n_iter, num_datapoints, total_params, total_params]), axis=0)
        fhat = fishers * total_params * param_vol / fisher_trace
        return fhat, fisher_trace


    def calc_all_grads(self):
        total_params = self.num_params
        inputs = tf.convert_to_tensor(self.inputs)
        n_classes = self.n_classes
        grads = np.zeros([len(inputs), len(inputs[0]), n_classes, total_params])
        for count,i in enumerate(tqdm(inputs, desc='Calculating Gradients')):
            outputs = self.model(i)
            grad = self.calc_jac(i, self.model)
            outputs = outputs.numpy()
            grad = tf.concat([tf.reshape(g, [g.shape[0], g.shape[1], tf.reduce_prod(g.shape[2:])]) for g in grad], axis=-1).numpy()
            grad[:,:,:] = grad[:,:,:] * np.sqrt(np.reshape(outputs, [*outputs.shape, 1]))
            grads[count] = grad
            self.reinitialize_model(self.model)
        grads = grads.reshape([len(inputs) * len(inputs[0]), n_classes, total_params])
        return tf.convert_to_tensor(grads)
    
    def calc_eigvals(self):
        fhat = self.fhat
        eigvals = np.linalg.eigvals(fhat).real
        return eigvals

    #def calc_effective_dimension(self, )
    
    def get_fishers(self):
        return self.fishers
    
    def get_fhat(self):
        return self.fhat
    
    def get_fisher_trace(self):
        return self.fisher_trace
    
    def get_grads(self):
        return self.grads


if __name__ == '__main__':
    model_func = lambda : Net(
        network_type='quantum',
        num_qumodes=2,
        n_layers=1,
        max_initial_weight=0.2,
        cutoff=2,
    )
    # model_func = lambda : keras.models.Sequential([QuantumLayer(
    #     n_qumodes=2,
    #     n_layers=1,
    #     max_initial_weight=0.1,
    #     cutoff_dim=2,
    # )])
    model = model_func()
    x_train = np.random.uniform(0, 1, size=[20, 4]) 
    print(model(x_train[0:1]))
    print(model.summary())
    fisher_obj = Calculate_Fisher_Information_Matrix(
        model_func,
        x_train,
        n_iter=5
    )
    print(fisher_obj.get_fhat())

 


# %%