import pandas as pd
from tqdm import tqdm
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from tensorflow import keras
import copy
from quantum_base_kerr import train_data, test_data, OPTIMIZER, LOSS_FUNCTION
from itertools import product
from concurrent.futures import ThreadPoolExecutor

validate_data = test_data



OPTIMIZER = tf.keras.optimizers.legacy.Adam()

df_kerr8 = pd.read_pickle('./dataframes/df_kerr8_all_weights.pkl', compression='xz') 

def get_maximum_amplitudes(df, 
                            metric='acc', 
                            enob_range=(1, 9), 
                            step_size=1, 
                            data=(train_data, validate_data), 
                            epoch=199):  
    df_final = copy.deepcopy(df)
    df_final[metric] = df[metric].apply(lambda x: x[-1])
    df_quantum = df_final[df_final['network_type']=='quantum']
    
    
    exp_folder = df_quantum['exp_folder'].unique()[0]

    max_a, max_a_squeezing = [], []
    for n in df_quantum['num_qumodes'].unique():
        for c in df_quantum[df_quantum['num_qumodes']==n]['cutoff'].unique():
            for nl in df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c)]['n_layers'].unique():
                exp_quantum = df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c) & (df_quantum['n_layers']==nl)]
                if len(exp_quantum) == 0:
                    continue
                else:
                    exp_quantum = exp_quantum[metric].idxmax()
                    print('For Number of Qumodes: ', n, ' Cutoff: ', int(c), ' Layers: ', nl)
                    print(f'Highest Accuracy Experiments\nQuantum: {exp_quantum}')
                    print(f'Quantum Val Acc, Loss: {df_quantum.loc[exp_quantum, "val_acc"][-1]}, {df_quantum.loc[exp_quantum, "loss"][-1]}')
                    model_quantum = df_quantum['model'][exp_quantum](
                        network_type='quantum',
                        num_qumodes=n,
                        n_layers=nl,
                        cutoff=int(c),
                        max_initial_weight=0.4489,
                        input_nl = None
                    )
                    
                    def get_max(model):
                        def check_string(name, comparisons):
                            return any([c in name for c in comparisons]) 
                        max_a = 0.
                        max_a_squeezing = 0.
                        for layer in model.layers:
                            if 'quantum_layer__multi_qunode' in layer.name:
                                quantum_weights = layer.get_weights() 
                                print(layer.weights, quantum_weights)
                                for w,val in zip(layer.weights, quantum_weights):
                                    print(w.name)
                                    if ('/a:' in w.name):
                                        if max_a < max(np.abs(val))[0]:
                                            max_a = max(np.abs(val))[0]
                                    if '/r:' in w.name:
                                        if max_a_squeezing < max(np.abs(val))[0]:
                                            max_a_squeezing = max(np.abs(val))[0]
                            elif 'sequential' in layer.name:
                                max_a_seq, max_a_squeezing_seq = get_max(layer)
                                if max_a < max_a_seq:
                                    max_a = max_a_seq
                                if max_a_squeezing < max_a_squeezing_seq:
                                    max_a_squeezing = max_a_squeezing_seq
                                
                            else:
                                continue
                        return max_a, max_a_squeezing 
                    
                    def load_weights(model, epoch, exp_folder, exp):
                        try:
                            model.load_weights(f'{exp_folder}{exp}/weights/weight.{epoch}.ckpt', by_name=False).expect_partial()
                        except:
                            try:
                                model.load_weights(f'{exp_folder}{exp}/weights/weight{epoch}.ckpt', by_name=False).expect_partial()
                            except:
                                try:
                                    model.load_weights(f'{exp_folder}{exp}/weights/weight.99.ckpt', by_name=False).expect_partial()
                                except:
                                    model.load_weights(f'{exp_folder}{exp}/weights/weight99.ckpt', by_name=False).expect_partial()
                        return model
                    
                    # Load the weights
                    model_quantum(tf.constant(data[0][0][0:2], dtype=tf.float32))
                    model_quantum = load_weights(model_quantum, epoch, exp_folder, exp_quantum)
                    max_a, max_a_squeezing = get_max(model_quantum)
                    print(f'Maximum Amplitude: {max_a}, Maximum Squeezing: {max_a_squeezing}')
                    df_quantum.loc[exp_quantum, 'max_amplitude'] = max_a
                    df_quantum.loc[exp_quantum, 'max_squeezing'] = max_a_squeezing
    return df_quantum

def convert_r_to_db(max_amplitude):
    return 10*np.log10(np.exp(2*max_amplitude))
    
if __name__ == '__main__':
    df_kerr8 = get_maximum_amplitudes(df_kerr8)
    print('Conversion to dB')
    df_kerr8['max_squeezing_db'] = df_kerr8['max_squeezing'].apply(convert_r_to_db)   
    df_kerr8.to_pickle('./dataframes/df_kerr8_all_weights_max_amplitudes.pkl', compression='xz')
    print(df_kerr8)
    


    
