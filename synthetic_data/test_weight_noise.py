import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import copy
from quantum_base_kerr import train_data, test_data, OPTIMIZER, LOSS_FUNCTION
validate_data = test_data

df_kerr8 = pd.read_pickle('./dataframes/df_kerr8.pkl', compression='xz') 

def plot_network_output_dist(df, metric='acc', noise_range=(0, -3), npoints=5, fname='network_certainty_dist', data=(train_data, validate_data), epoch=99):  
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
                        max_initial_weight=0.5
                    )

                    model_classical = df_classical['model'][exp_classical](
                        network_type='classical',
                        num_qumodes=n,
                        n_layers=nl,
                        cutoff=-1,
                        max_initial_weight=0.5
                    )

                    def add_noise(weights, noise):
                        return weights + tf.random.normal(tf.shape(weights), stddev=noise)
                    
                    def test_model(model, noise):
                        model.set_weights([add_noise(w, noise) for w in model.get_weights()])
                        return model
                        
                    def evaluate(model, noise, x, y):
                        model = test_model(model, noise)
                        return model.evaluate(x, y)
                    
                    def build_df(model, noise, plot_df=plot_df):
                        for i in range(5):
                            df_temp = pd.DataFrame(columns=plot_df.columns)
                            acc, loss = evaluate(model, s, train_data[0], train_data[1])
                            val_acc, val_loss = evaluate(model, s, validate_data[0], validate_data[1])
                            if model.network_type == 'quantum':
                                enob_phase = np.log2(1 + (2*np.pi/s))
                                enob_amp = np.log2(1 + (model.quantum_layer.max_initial_weight/s))
                                proportion_amp = 2/(2*n+5)
                                df_temp['enob'] =  proportion_amp*enob_amp + (1-proportion_amp)*enob_phase
                            elif model.network_type == 'classical':
                                df_temp.loc['enob'] = np.log2(1 + 2/(s))
                            else:
                                raise Exception('Network type not recognized')
                            df_temp['cutoff'] = c
                            df_temp['epoch'] = epoch
                            df_temp['noise'] = s
                            df_temp['n_layers'] = nl
                            df_temp['num_qumodes'] = n
                            df_temp['acc'] = acc
                            df_temp['loss'] = loss
                            df_temp['val_acc'] = val_acc
                            df_temp['val_loss'] = val_loss
                            df_temp['sample'] = i
                            df_temp['network_type'] = model.network_type
                            df_temp['exp_folder'] = exp_folder
                            df_temp.reset_index(inplace=True)
                            print(df_temp, plot_df)
                            plot_df.loc[len(plot_df)] = df_temp
                        return plot_df
                    
                    model_quantum.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"]) 
                    model_classical.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])

                    for s in np.logspace(noise_range[0], noise_range[1], npoints): 
                        try:
                            model_quantum.load_weights(f'{exp_folder}{exp_quantum}/weights/weight.{epoch}.ckpt', by_name=False).expect_partial()
                        except:
                            model_quantum.load_weights(f'{exp_folder}{exp_quantum}/weights/weight{epoch}.ckpt', by_name=False).expect_partial()
                        try:
                            model_classical.load_weights(f'{exp_folder}{exp_classical}/weights/weight.{epoch}.ckpt', by_name=False).expect_partial()
                        except:
                            model_classical.load_weights(f'{exp_folder}{exp_classical}/weights/weight{epoch}.ckpt', by_name=False).expect_partial()
                        
                        model_quantum(validate_data[0][0:1])
                        model_classical(validate_data[0][0:1])

                        #plot_df = build_df(model_quantum, s)
                        plot_df = build_df(model_classical, s)                        
                                   
    return plot_df
df_kerr8 = df_kerr8[(df_kerr8['num_qumodes']==2) & (df_kerr8['n_layers']==1) & ((df_kerr8['cutoff']==11) | (df_kerr8['cutoff'] == -1))]                         
noise_df = plot_network_output_dist(df_kerr8, metric='acc', fname='network_certainty_dist')
print(noise_df)

pd.to_pickle(noise_df, './dataframes/noise_df.pkl', compression='xz')
