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

def generate_noise_dataframe(df, metric='acc', noise_range=(0.4, -3), npoints=50, data=(train_data, validate_data), epoch=199):  
    df_final = copy.deepcopy(df)
    df_final[metric] = df[metric].apply(lambda x: x[-1])
    df_quantum = df_final[df_final['network_type']=='quantum']
    df_quantum = df_quantum[df_quantum['num_qumodes']==2]
    df_classical = df_final[df_final['network_type']=='classical']
    df_classical = df_classical[df_classical['num_qumodes']==2]
    train_data, validate_data = data

    plot_df = pd.DataFrame(columns=df_final.columns)
    plot_df['noise'] = np.NaN
    plot_df.drop(['acc', 'val_acc', 'loss', 'val_loss', 'model', 'iteration', 'num_params'], axis=1, inplace=True)
    plot_df['acc'] = np.NaN
    plot_df['loss'] = np.NaN
    plot_df['val_acc'] = np.NaN
    plot_df['val_loss'] = np.NaN
    plot_df['sample'] = np.NaN
    plot_df['enob'] = np.NaN
    plot_df.reset_index(inplace=True)
    plot_df.drop(['index'], inplace=True, axis=1)
    
    exp_folder = df_quantum['exp_folder'].unique()[0]
    for n in df_quantum['num_qumodes'].unique():
        for c in df_quantum[df_quantum['num_qumodes']==n]['cutoff'].unique():
            for nl in df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c)]['n_layers'].unique():
                exp_quantum = df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c) & (df_quantum['n_layers']==nl)]
                exp_classical = df_classical.loc[(df_classical['num_qumodes']==n) & (df_classical['n_layers']==nl)]
                if (len(exp_quantum) == 0) or (len(exp_classical) == 0):
                    continue
                else:
                    exp_quantum = exp_quantum[metric].idxmax()
                    exp_classical = exp_classical[metric].idxmax()
                    print('For Number of Qumodes: ', n, ' Cutoff: ', int(c), ' Layers: ', nl)
                    print(f'Highest Accuracy Experiments\nQuantum: {exp_quantum}\nClassical: {exp_classical}')
                    print(f'Quantum Val Acc, Acc: {df_quantum.loc[exp_quantum, "val_acc"][-1]}, {df_quantum.loc[exp_quantum, "loss"][-1]}')
                    print(f'Classical Val Acc, Acc: {df_classical.loc[exp_classical, "val_acc"][-1]}, {df_classical.loc[exp_classical, "loss"][-1]}')
                    model_quantum = df_quantum['model'][exp_quantum](
                        network_type='quantum',
                        num_qumodes=n,
                        n_layers=nl,
                        cutoff=int(c),
                        max_initial_weight=None,
                    )

                    model_classical = df_classical['model'][exp_classical](
                        network_type='classical',
                        num_qumodes=n,
                        n_layers=nl,
                        cutoff=-1,
                        max_initial_weight=0.5,
                    )

                    def add_noise(weights, noise):
                        return weights + tf.random.normal(tf.shape(weights), stddev=noise)
                    
                    def test_model(model, noise):
                        model.set_weights([add_noise(w, noise) for w in model.get_weights()])
                        return model
                        
                    def evaluate(model, noise, x, y):
                        model = test_model(model, noise)
                        return model.evaluate(x, y)
                    
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
                    
                    def build_df(model, noise, plot_df=plot_df):
                        weights = model.get_weights()
                        for i in range(10):
                            df_temp = pd.DataFrame(columns=plot_df.columns)
                            model.set_weights(weights)
                            loss, acc = evaluate(model, noise, train_data[0], train_data[1])
                            val_loss, val_acc = evaluate(model, noise, validate_data[0], validate_data[1])
                            if model.network_type == 'quantum':
                                enob_phase = np.log2(1 + (2*np.pi/s))
                                enob_amp = np.log2(1 + (model.quantum_layer.max_initial_weight/s))
                                enob_classical = np.log2(1+2/s)
                                proportion_amp = 2/(2*n + 5) 
                                df_temp.loc[0, 'enob'] =  proportion_amp*enob_amp + (1-proportion_amp)*enob_phase
                            elif model.network_type == 'classical':
                                df_temp.loc[0, 'enob'] = np.log2(1 + 2/(s))
                            else:
                                raise Exception('Network type not recognized')
                            df_temp.loc[0, 'cutoff'] = c
                            df_temp.loc[0, 'epoch'] = epoch
                            df_temp.loc[0, 'noise'] = s
                            df_temp.loc[0, 'n_layers'] = nl
                            df_temp.loc[0, 'num_qumodes'] = n
                            df_temp.loc[0, 'acc'] = acc
                            df_temp.loc[0, 'loss'] = loss
                            df_temp.loc[0, 'val_acc'] = val_acc
                            df_temp.loc[0, 'val_loss'] = val_loss
                            df_temp.loc[0, 'sample'] = i
                            df_temp.loc[0, 'network_type'] = model.network_type
                            df_temp.loc[0, 'exp_folder'] = exp_folder
                            plot_df.loc[len(plot_df)] = df_temp.loc[0]

                        return plot_df
                    
                    model_quantum.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"]) 
                    model_classical.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])

                    for s in np.logspace(noise_range[0], noise_range[1], npoints): 
                        model_quantum = load_weights(model_quantum, epoch, exp_folder, exp_quantum)
                        model_classical = load_weights(model_classical, epoch, exp_folder, exp_classical)
                        
                        model_quantum(validate_data[0][0:1])
                        model_classical(validate_data[0][0:1])

                        plot_df = build_df(model_quantum, s)
                        plot_df = build_df(model_classical, s)                        
                                   
    return plot_df

def generate_enob_dataframe(df, 
                            metric='acc', 
                            enob_range=(0.5, 10), 
                            step_size=0.5, 
                            data=(train_data, validate_data), 
                            epoch=195):  
    df_final = copy.deepcopy(df)
    df_final[metric] = df[metric].apply(lambda x: x[-1])
    df_quantum = df_final[df_final['network_type']=='quantum']
    df_classical = df_final[df_final['network_type']=='classical']
    train_data, validate_data = data

    plot_df = pd.DataFrame(columns=df_final.columns)
    plot_df['noise'] = np.NaN
    plot_df.drop(['acc', 'val_acc', 'loss', 'val_loss', 'model', 'iteration', 'num_params'], axis=1, inplace=True)
    plot_df['acc'] = np.NaN
    plot_df['loss'] = np.NaN
    plot_df['val_acc'] = np.NaN
    plot_df['val_loss'] = np.NaN
    plot_df['sample'] = np.NaN
    plot_df['enob'] = np.NaN
    plot_df.reset_index(inplace=True)
    plot_df.drop(['index'], inplace=True, axis=1)

    exp_folder_quantum = df_quantum['exp_folder'].unique()[0]
    exp_folder_classical = df_classical['exp_folder'].unique()[0]
    
    for n in df_quantum['num_qumodes'].unique():
        for c in df_quantum[df_quantum['num_qumodes']==n]['cutoff'].unique():
            for nl in df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c)]['n_layers'].unique():
                exp_quantum = df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c) & (df_quantum['n_layers']==nl)]
                exp_classical = df_classical.loc[(df_classical['num_qumodes']==n) & (df_classical['n_layers']==nl)]
                if (len(exp_quantum) == 0) or (len(exp_classical) == 0):
                    continue
                else:
                    exp_quantum = exp_quantum[exp_quantum[metric] == exp_quantum[metric].median()].index[0]
                    exp_classical = exp_classical[exp_classical[metric] == exp_classical[metric].median()].index[0]
                    print('For Number of Qumodes: ', n, ' Cutoff: ', int(c), ' Layers: ', nl)
                    print(f'Highest Accuracy Experiments\nQuantum: {exp_quantum}\nClassical: {exp_classical}')
                    print(f'Quantum Val Acc, Acc: {df_quantum.loc[exp_quantum, "val_acc"][-1]}, {df_quantum.loc[exp_quantum, "loss"][-1]}')
                    print(f'Classical Val Acc, Acc: {df_classical.loc[exp_classical, "val_acc"][-1]}, {df_classical.loc[exp_classical, "loss"][-1]}')
                    model_quantum = df_quantum['model'][exp_quantum](
                        network_type='quantum',
                        num_qumodes=n,
                        n_layers=nl,
                        cutoff=int(c),
                        max_initial_weight=None,
                    )

                    model_classical = df_classical['model'][exp_classical](
                        network_type='classical',
                        num_qumodes=n,
                        n_layers=nl,
                        cutoff=-1,
                        max_initial_weight=0.5,
                    )
                    
                    def get_noise(model, enob, amplitude_noise=True, phase_noise=True):
                        weights_noise = []
                        max_a = 0
                        for layer in model.layers:
                            if 'quantum_layer__multi_qunode' in layer.name:
                                quantum_weights = layer.get_weights()
                                for w,val in zip(layer.weights, quantum_weights):
                                    if ('/r:' in w.name) or ('/a:' in w.name):
                                        try:
                                            if max_a < max(np.abs(val))[0]:
                                                max_a = max(np.abs(val))[0]
                                        except ValueError:
                                            for v in val:
                                                if max_a < max(np.abs(v)):
                                                    max_a = max(np.abs(v))
                                for w, val in zip(layer.weights, quantum_weights):
                                    if (('/r:' in w.name) or ('/a:' in w.name)) and amplitude_noise:
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=max_a/(2**enob-1)))
                                    elif phase_noise:
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=2*np.pi/(2**enob-1)))
                            elif 'sequential' in layer.name:
                                sequential_weights = get_noise(layer, enob)
                                for w in sequential_weights:
                                    weights_noise.append(w)
                            elif 'dense' in layer.name:
                                dense_weights = layer.get_weights()
                                for w in dense_weights:
                                    weights_noise.append(tf.random.normal(tf.shape(w), stddev=2/(2**enob-1)))
                            else:
                                continue
                        return weights_noise                                   
                        
                    def evaluate(model, enob, x, y):
                        noise = get_noise(model, enob)
                        model.set_weights([w + n for w, n in zip(model.get_weights(), noise)])
                        return model.evaluate(x, y, verbose=0)
                    
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
                    
                    def build_df(model, enob, exp_folder, plot_df=plot_df):
                        weights = model.get_weights()
                        for i in range(10):
                            df_temp = pd.DataFrame(columns=plot_df.columns)
                            model.set_weights(weights)
                            loss, acc = evaluate(model, enob, train_data[0], train_data[1])
                            val_loss, val_acc = evaluate(model, enob, validate_data[0], validate_data[1])

                            df_temp.loc[0, 'enob'] = enob
                            df_temp.loc[0, 'cutoff'] = c
                            df_temp.loc[0, 'epoch'] = epoch
                            df_temp.loc[0, 'n_layers'] = nl
                            df_temp.loc[0, 'num_qumodes'] = n
                            df_temp.loc[0, 'acc'] = acc
                            df_temp.loc[0, 'loss'] = loss
                            df_temp.loc[0, 'val_acc'] = val_acc
                            df_temp.loc[0, 'val_loss'] = val_loss
                            df_temp.loc[0, 'sample'] = i
                            df_temp.loc[0, 'network_type'] = model.network_type
                            df_temp.loc[0, 'exp_folder'] = exp_folder
                            plot_df.loc[len(plot_df)] = df_temp.loc[0]

                        return plot_df
                    
                    model_quantum.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"]) 
                    model_classical.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])

                    for e in tqdm(np.arange(enob_range[0], enob_range[1]+step_size, step_size)): 
                        model_quantum = load_weights(model_quantum, epoch, exp_folder_quantum, exp_quantum)
                        model_classical = load_weights(model_classical, epoch, exp_folder_classical, exp_classical)
                        
                        # model_quantum(validate_data[0][0:1])
                        model_classical(validate_data[0][0:1])

                        # plot_df = build_df(model_quantum, e, exp_folder_quantum)
                        plot_df = build_df(model_classical, e, exp_folder_classical)
                                   
    return plot_df

def generate_enob_dataframe_amp_phase(df, 
                            metric='acc', 
                            enob_range=(0.5, 10), 
                            step_size=0.5, 
                            data=(train_data, validate_data), 
                            epoch=199):  
    df_final = copy.deepcopy(df)
    df_final[metric] = df[metric].apply(lambda x: x[-1])
    df_quantum = df_final[df_final['network_type']=='quantum']
    train_data, validate_data = data

    plot_df = pd.DataFrame(columns=df_final.columns)
    plot_df['noise'] = np.NaN
    plot_df.drop(['acc', 'val_acc', 'loss', 'val_loss', 'model', 'iteration', 'num_params'], axis=1, inplace=True)
    plot_df['acc'] = np.NaN
    plot_df['loss'] = np.NaN
    plot_df['val_acc'] = np.NaN
    plot_df['val_loss'] = np.NaN
    plot_df['sample'] = np.NaN
    plot_df['amplitude_enob'] = np.NaN
    plot_df['phase_enob'] = np.NaN
    plot_df.reset_index(inplace=True)
    plot_df.drop(['index'], inplace=True, axis=1)
    
    exp_folder = df_quantum['exp_folder'].unique()[0]
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
                    print(f'Quantum Val Acc, Acc: {df_quantum.loc[exp_quantum, "val_acc"][-1]}, {df_quantum.loc[exp_quantum, "loss"][-1]}')
                    model_quantum = df_quantum['model'][exp_quantum](
                        network_type='quantum',
                        num_qumodes=n,
                        n_layers=nl,
                        cutoff=int(c),
                        max_initial_weight=None,
                    )
                    
                    def get_noise(model, amplitude_enob, phase_enob):
                        weights_noise = []
                        max_a = 0
                        for layer in model.layers:
                            if 'quantum_layer__multi_qunode' in layer.name:
                                quantum_weights = layer.get_weights() 
                                for w,val in zip(layer.weights, quantum_weights):
                                    if ('/r:' in w.name) or ('/a:' in w.name):
                                        if max_a < max(np.abs(val))[0]:
                                            max_a = max(np.abs(val))[0]
                                for w, val in zip(layer.weights, quantum_weights):
                                    if (('/r:' in w.name) or ('/a:' in w.name)):
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=max_a/(2**amplitude_enob-1)))
                                    else:
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=2*np.pi/(2**phase_enob-1)))
                            elif 'sequential' in layer.name:
                                sequential_weights = get_noise(layer, amplitude_enob, phase_enob)
                                for w in sequential_weights:
                                    weights_noise.append(w)
                            elif 'dense' in layer.name:
                                dense_weights = layer.get_weights()
                                for w in dense_weights:
                                    weights_noise.append(tf.random.normal(tf.shape(w), stddev=2/(2**16-1)))
                            else:
                                continue
                        return weights_noise                                   
                        
                    def evaluate(model, amplitude_enob, phase_enob, x, y):
                        noise = get_noise(model, amplitude_enob, phase_enob)
                        model.set_weights([w + n for w, n in zip(model.get_weights(), noise)])
                        return model.evaluate(x, y, verbose=0)
                    
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
                    
                    def build_df(model, amplitude_enob, phase_enob, plot_df=plot_df):
                        weights = model.get_weights()
                        for i in range(5):
                            df_temp = pd.DataFrame(columns=plot_df.columns)
                            model.set_weights(weights)
                            loss, acc = evaluate(model, amplitude_enob, phase_enob, train_data[0], train_data[1])
                            val_loss, val_acc = evaluate(model, amplitude_enob, phase_enob, validate_data[0], validate_data[1])

                            df_temp.loc[0, 'amplitude_enob'] = amplitude_enob
                            df_temp.loc[0, 'phase_enob'] = phase_enob
                            df_temp.loc[0, 'cutoff'] = c
                            df_temp.loc[0, 'epoch'] = epoch
                            df_temp.loc[0, 'n_layers'] = nl
                            df_temp.loc[0, 'num_qumodes'] = n
                            df_temp.loc[0, 'acc'] = acc
                            df_temp.loc[0, 'loss'] = loss
                            df_temp.loc[0, 'val_acc'] = val_acc
                            df_temp.loc[0, 'val_loss'] = val_loss
                            df_temp.loc[0, 'sample'] = i
                            df_temp.loc[0, 'network_type'] = model.network_type
                            df_temp.loc[0, 'exp_folder'] = exp_folder
                            plot_df.loc[len(plot_df)] = df_temp.loc[0]

                        return plot_df
                    
                    model_quantum.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"]) 

                    for ae in tqdm(np.arange(enob_range[0], enob_range[1]+step_size, step_size)): 
                        for pe in tqdm(np.arange(enob_range[0], enob_range[1]+step_size, step_size)): 
                            model_quantum = load_weights(model_quantum, epoch, exp_folder, exp_quantum)
                            
                            model_quantum(validate_data[0][0:1])

                            plot_df = build_df(model_quantum, ae, pe)
                                   
    return plot_df


def generate_enob_dataframe_gate_based(df, 
                            metric='acc', 
                            enob_range=(1, 9), 
                            step_size=1, 
                            data=(train_data, validate_data), 
                            epoch=199):  
    df_final = copy.deepcopy(df)
    df_final[metric] = df[metric].apply(lambda x: x[-1])
    df_quantum = df_final[df_final['network_type']=='quantum']
    train_data, validate_data = data

    plot_df = pd.DataFrame(columns=df_final.columns)
    plot_df.drop(['acc', 'val_acc', 'loss', 'val_loss', 'model', 'iteration', 'num_params'], axis=1, inplace=True)
    plot_df['val_acc'] = np.NaN
    plot_df['val_loss'] = np.NaN
    plot_df['sample'] = np.NaN
    plot_df['squeezing_enob'] = np.NaN
    plot_df['displacement_enob'] = np.NaN
    plot_df['kerr_enob'] = np.NaN
    plot_df['interferometer_enob'] = np.NaN
    plot_df.reset_index(inplace=True)
    plot_df.drop(['index'], inplace=True, axis=1)
    
    exp_folder = df_quantum['exp_folder'].unique()[0]
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
                    
                    def get_noise(model, squeezing_enob, kerr_enob, interferometer_enob, displacement_enob):
                        def check_string(name, comparisons):
                            return any([c in name for c in comparisons]) 
                        squeezing_parameter_names = ['/r:', '/phi_r:']
                        kerr_parameter_names = ['/k:']
                        interferometer_parameter_names = ['/theta_1:', '/theta_2:', '/phi_1:', '/phi_2:', '/varphi_1:', '/varphi_2:']
                        displacement_parameter_names = ['/a:', '/phi_a:']
                        weights_noise = []
                        max_a = 0
                        for layer in model.layers:
                            if 'quantum_layer__multi_qunode' in layer.name:
                                quantum_weights = layer.get_weights() 
                                for w,val in zip(layer.weights, quantum_weights):
                                    if ('/r:' in w.name) or ('/a:' in w.name):
                                        if max_a < max(np.abs(val))[0]:
                                            max_a = max(np.abs(val))[0]
                                for w, val in zip(layer.weights, quantum_weights):
                                    if (squeezing_parameter_names[1] in w.name):
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=2*np.pi/(2**squeezing_enob-1)))
                                    elif (squeezing_parameter_names[0] in w.name):
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=max_a/(2**squeezing_enob-1)))
                                    elif check_string(w.name, kerr_parameter_names):
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=2*np.pi/(2**kerr_enob-1)))
                                    elif check_string(w.name, interferometer_parameter_names):
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=2*np.pi/(2**interferometer_enob-1)))
                                    elif (displacement_parameter_names[0] in w.name):
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=max_a/(2**displacement_enob-1)))
                                    elif  (displacement_parameter_names[1] in w.name):
                                        weights_noise.append(tf.random.normal(tf.shape(w), stddev=2*np.pi/(2**displacement_enob-1)))
                            elif 'sequential' in layer.name:
                                sequential_weights = get_noise(layer, squeezing_enob, kerr_enob, interferometer_enob, displacement_enob)
                                for w in sequential_weights:
                                    weights_noise.append(w)
                            elif 'dense' in layer.name:
                                dense_weights = layer.get_weights()
                                for w in dense_weights:
                                    weights_noise.append(tf.random.normal(tf.shape(w), stddev=2/(2**32-1)))
                            else:
                                continue
                        return weights_noise                                   
                        
                    def evaluate(model, squeezing_enob, kerr_enob, interferometer_enob, displacement_enob, x, y):
                        noise = get_noise(model, squeezing_enob, kerr_enob, interferometer_enob, displacement_enob)
                        model.set_weights([w + n for w, n in zip(model.get_weights(), noise)])
                        return model.evaluate(x, y, verbose=0)
                    
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
                    
                    def build_df(model, squeezing_enob, kerr_enob, interferometer_enob, displacement_enob, columns):
                        plot_df = pd.DataFrame(columns=columns)
                        weights = model.get_weights()
                        for i in range(10):
                            df_temp = pd.DataFrame(columns=plot_df.columns)
                            model.set_weights(weights)
                            val_loss, val_acc = evaluate(model, squeezing_enob, kerr_enob, interferometer_enob, displacement_enob, validate_data[0], validate_data[1])

                            df_temp.loc[0, 'squeezing_enob'] = squeezing_enob
                            df_temp.loc[0, 'kerr_enob'] = kerr_enob
                            df_temp.loc[0, 'interferometer_enob'] = interferometer_enob
                            df_temp.loc[0, 'displacement_enob'] = displacement_enob
                            df_temp.loc[0, 'cutoff'] = c
                            df_temp.loc[0, 'epoch'] = epoch
                            df_temp.loc[0, 'n_layers'] = nl
                            df_temp.loc[0, 'num_qumodes'] = n
                            df_temp.loc[0, 'val_acc'] = val_acc
                            df_temp.loc[0, 'val_loss'] = val_loss
                            df_temp.loc[0, 'sample'] = i
                            df_temp.loc[0, 'network_type'] = model.network_type
                            df_temp.loc[0, 'exp_folder'] = exp_folder
                            plot_df.loc[len(plot_df)] = df_temp.loc[0]

                        return plot_df
                    
                    def process_combination(model_quantum, params, columns=plot_df.columns):
                        se, ke, ie, de = params
                        model_quantum = load_weights(model_quantum, epoch, exp_folder, exp_quantum)
                        model_quantum(validate_data[0][0:1])
                        return build_df(model_quantum, se, ke, ie, de, columns=columns)
                    
                    model_quantum.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"]) 

                    enobs = np.arange(enob_range[0], enob_range[1]+step_size, step_size)


                    model_trained = load_weights(model_quantum, epoch, exp_folder, exp_quantum)
                    
                    model_trained.evaluate(*validate_data)

                    print(f'Total experiment time', 4*len(enobs)*10*15/60, 'minutes')

                    params_default = (32, 32, 32, 32)
                    for i in tqdm(range(len(params_default))):
                        for enob in tqdm(enobs):
                            params = list(params_default)
                            params[i] = enob
                            plot_df = pd.concat([plot_df, process_combination(model_quantum, params)], axis=0)

                    
                    
                        
                                   
    return plot_df





if __name__ == '__main__':
    df_kerr8 = df_kerr8.loc[[1, 358]]
    df_kerr8 = df_kerr8[(df_kerr8['num_qumodes']==2) & (df_kerr8['n_layers']==1) & ((df_kerr8['cutoff']==11) | (df_kerr8['cutoff'] == -1))]                         
    quantum_noise_df = generate_enob_dataframe_gate_based(df_kerr8, step_size=0.5, enob_range=(0.5, 10))
    pd.to_pickle(quantum_noise_df, './dataframes/quantum_noise_df.pkl')
