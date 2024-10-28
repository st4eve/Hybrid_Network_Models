from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import seaborn as sns
import glob


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from tensorflow import keras
import copy
from quantum_base_kerr import train_data, test_data, OPTIMIZER, LOSS_FUNCTION, Net

validate_data = test_data

palette = sns.color_palette('pastel')
colors = palette.as_hex()
sns.set_palette(palette)


sns.set_style('ticks')
sns.set_context("paper")

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["axes.linewidth"] = 0.5

OPTIMIZER = tf.keras.optimizers.legacy.Adam()

# df_kerr8 = pd.read_pickle('./dataframes/df_kerr8_all_weights.pkl', compression='xz') 
df_kerr8 = pd.read_pickle('./dataframes/df_kerr8.pkl', compression='xz')


def get_maximum_amplitudes(df, 
                            metric='acc', 
                            step_size=1, 
                            data=(train_data, validate_data), 
                            epoch=199):  

                            
    def get_max(model):
        max_a = 0.
        max_a_squeezing = 0.
        for layer in model.layers:
            if 'quantum_layer__multi_qunode' in layer.name:
                quantum_weights = layer.get_weights() 
                for w,val in zip(layer.weights, quantum_weights):
                    if ('/a:' in w.name):
                        if max_a < np.max(np.abs(val)):
                            max_a = np.max(np.abs(val))
                    if '/r:' in w.name:
                        if max_a_squeezing < np.max(np.abs(val)):
                            max_a_squeezing = np.max(np.abs(val))
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
                    try:
                        model.load_weights(f'{exp_folder}{exp}/weights/weight99.ckpt', by_name=False).expect_partial()
                    except:
                        files = glob.glob(f'{exp_folder}{exp}/weights/*.ckpt')
                        model.load_weights(files[-1], by_name=False).expect_partial()
        return model
    
    df_final = copy.deepcopy(df)
    df_final[metric] = df[metric].apply(lambda x: x[-1])
    df_quantum = df_final[df_final['network_type']=='quantum']


    df_quantum.loc[:, 'max_amplitude'] = np.nan
    df_quantum.loc[:, 'max_squeezing'] = np.nan
    


    for n in df_quantum['num_qumodes'].unique():
        for c in df_quantum[df_quantum['num_qumodes']==n]['cutoff'].unique():
            for nl in df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c)]['n_layers'].unique():
                exp_quantum = df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c) & (df_quantum['n_layers']==nl)]
                if len(exp_quantum) == 0:
                    continue
                else:
                    for exp in exp_quantum.index:
                        try:
                            model_quantum = df_quantum.loc[exp, 'model'](
                                network_type='quantum',
                                num_qumodes=int(n),
                                n_layers=int(nl),
                                cutoff=int(c),
                                max_initial_weight=0.4489,
                                input_nl = None
                            )
                        except:
                            print(f'Error in loading model for {exp}')
                            continue
                        
                                               
                        # Load the weights
                        model_quantum(tf.constant(data[0][0][0:2], dtype=tf.float32))
                        try:
                            model_quantum = load_weights(model_quantum, epoch, df_quantum.loc[exp, 'exp_folder'], exp)
                            max_a, max_a_squeezing = get_max(model_quantum)
                            df_quantum.loc[exp, 'max_amplitude'] = max_a
                            df_quantum.loc[exp, 'max_squeezing'] = max_a_squeezing
                        except:
                            print(f'Error in loading weights for {exp}')
                            continue
    return df_quantum

def minimum_variance(r):
    return np.exp(-2*r)

def convert_to_db(V):
    return 10*np.log10(V)

def convert_r_to_db(max_amplitude):
    return convert_to_db(minimum_variance(max_amplitude))

def calculate_variance_with_loss(r, eta):
    return eta*minimum_variance(r) + 1 - eta

def calculate_loss_from_variance(r1, r2):
    return (minimum_variance(r2) -1) / (minimum_variance(r1) - 1)

def df_loss_calculation(df):
    df_loss = df.copy()
    df_loss['max_squeezing_db'] = df_loss['max_squeezing'].apply(convert_r_to_db)
    # Set eta to 1.0 for cutoff == 11
    df_loss.loc[df_loss['cutoff'] == 11, 'eta'] = 1.0
    
    
    # Filter for cutoff == 11 and cutoff != 11
    df_11 = df_loss[df_loss['cutoff'] == 11]
    idx = df_11.groupby(['num_qumodes', 'n_layers'])['val_acc'].idxmax()
    df_11 = df_11.loc[idx].reset_index()
    df_others = df_loss[df_loss['cutoff'] != 11]
    df_others = df_others[df_others['num_qumodes'].isin(df_11['num_qumodes'].unique())]
    
    
    
    # Merge the dataframes on 'num_qumodes' and 'n_layers' to align r1 and r2
    merged_df = pd.merge(df_11, df_others, on=['num_qumodes', 'n_layers'], suffixes=('_r1', '_r2'))

    
    # Calculate loss from variance
    merged_df['eta'] = merged_df.apply(
        lambda row: calculate_loss_from_variance(row['max_squeezing_r1'], row['max_squeezing_r2']), axis=1
    )
    merged_df['accuracy_difference'] = merged_df['val_acc_r2'] - merged_df['val_acc_r1']
    merged_df['photonic_loss'] = 10*np.log10(merged_df['eta'])
    merged_df.drop(columns=['eta_r2'], inplace=True)
    return merged_df
    
if __name__ == '__main__':
    try:
        # raise Exception()
        df_kerr8 = pd.read_pickle('./dataframes/df_kerr8_all_weights_max_amplitudes.pkl', compression='xz')
        df_kerr8.dropna(inplace=True)
    except:
        df_kerr8 = get_maximum_amplitudes(df_kerr8)
        df_kerr8['max_squeezing_db'] = df_kerr8['max_squeezing'].apply(convert_r_to_db)   
        df_kerr8.dropna(inplace=True)
        df_kerr8.to_pickle('./dataframes/df_kerr8_all_weights_max_amplitudes.pkl', compression='xz')

    
    df_kerr8['val_acc'] = df_kerr8['val_acc'].apply(lambda x: x[-1])
    df_kerr8['val_loss'] = df_kerr8['val_loss'].apply(lambda x: x[-1])
    groups  = ['num_qumodes', 'cutoff', 'n_layers']
    columns = ['val_acc', 'val_loss', 'max_amplitude', 'max_squeezing']
    df_kerr8_mean = df_kerr8.groupby(groups, group_keys=True)[columns].mean()
    # df_kerr8_mean = df_kerr8[groups + columns]
    df_kerr8_mean.reset_index(inplace=True)

    print(df_kerr8_mean)


    df_kerr8_loss = df_kerr8_mean.copy()
    
    df_kerr8_loss['eta'] = np.NaN

    df_kerr8_loss = df_loss_calculation(df_kerr8_loss)

    
    print(df_kerr8_loss[df_kerr8_loss['eta'] > 1.0])

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    # ax2 = ax.twiny()
    sns.lineplot(data=df_kerr8_loss, x='photonic_loss', y='accuracy_difference', hue='n_layers', style='num_qumodes', markers=True, palette=palette[2:], ax=ax)
    # sns.lineplot(data=df_kerr8_loss, x='max_squeezing_db_r2', y='accuracy_difference', hue='n_layers', style='num_qumodes', markers=True, palette=palette[2:], ax=ax2)
    plt.xlabel('Photonic Loss (dB)')
    plt.ylabel('Average Accuracy Difference')
    plt.savefig('./figures/photonic_loss_vs_accuracy_difference.png', dpi=300)
    plt.savefig('./figures/photonic_loss_vs_accuracy_difference.pdf', dpi=300)
    plt.show()




    
    
    


    
