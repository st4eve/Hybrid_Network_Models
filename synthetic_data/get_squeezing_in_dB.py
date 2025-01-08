from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.special
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import seaborn as sns
import glob
import scipy


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

def convert_db_to_r(db):
    return -np.log(10**(-db/10))/2

def calculate_variance_with_loss(r, eta):
    return eta*minimum_variance(r) + 1 - eta

def calculate_loss_from_variance(r1, r2):
    return (minimum_variance(r2) - 1) / (minimum_variance(r1) - 1)

def calculate_initial_variance_from_loss(r, eta):
    return (eta - 1 + minimum_variance(r)) / eta

def df_loss_calculation(df):
    df_loss = df.copy()
    df_loss['max_squeezing_db'] = df_loss['max_squeezing'].apply(convert_r_to_db)
    # Set eta to 1.0 for cutoff == 11
    df_loss.loc[df_loss['cutoff'] == 11, 'eta'] = 1.0
    
    
    # Filter for cutoff == 11 and cutoff != 11
    df_11 = df_loss[df_loss['cutoff'] == 11]
    idx = df_11.groupby(['num_qumodes', 'n_layers'])['val_acc'].idxmax()
    df_11 = df_11.loc[idx].reset_index()
    # df_others = df_loss[df_loss['cutoff'] != 11]
    df_others = df_loss.copy()
    df_others = df_others[df_others['num_qumodes'].isin(df_11['num_qumodes'].unique())]
    
    
    
    # Merge the dataframes on 'num_qumodes' and 'n_layers' to align r1 and r2
    merged_df = pd.merge(df_11, df_others, on=['num_qumodes', 'n_layers'], suffixes=('_r1', '_r2'))

    
    # Calculate loss from variance
    merged_df['eta'] = merged_df.apply(
        lambda row: calculate_loss_from_variance(row['max_squeezing_r1'], row['max_squeezing_r2']), axis=1
    )
    merged_df['accuracy_difference'] = merged_df['val_acc_r2'] - merged_df['val_acc_r1']
    merged_df['photonic_loss'] = 10*np.log10(merged_df['eta'])
    merged_df['eta_r2_dB'] = 10*np.log10(merged_df['eta_r2'])
    merged_df.drop(columns=['eta_r2'], inplace=True)
    return merged_df

def df_loss_calculation_v2(df, max_squeezing):
    """This version of the loss calculation function uses the maximum squeezing
       as the zero loss point. The loss is then calculated using the ratio of maximum
       squeezing seein the function calculate_loss_from_variance.    

    Args:
        df (pandas.Dataframe): Dataframe containing the maximum squeezing values.
    """

    df_loss = df.copy()

    
    max_squeezing_db = convert_r_to_db(max_squeezing)
    df_loss['max_squeezing_db'] = -df_loss['max_squeezing'].apply(convert_r_to_db)


    df_loss['eta'] = df_loss['max_squeezing'].apply(lambda x: calculate_loss_from_variance(max_squeezing, x))
    df_loss['photonic_loss'] = 10*np.log10(df_loss['eta'])

    if 'max_squeezing_std' in df_loss.columns:
        df_loss['eta_std'] = np.abs(2*minimum_variance(df_loss['max_squeezing'])/(minimum_variance(max_squeezing)-1)) * df_loss['max_squeezing_std']
        df_loss['photonic_loss_std'] = 10*1/(np.log(10)) * df_loss['eta_std'] / df_loss['eta']
    return df_loss


def plot_coloured_cutoff_violin(df, ax):

    def colour_cutoffs(ax, df):
        def get_linecollections(ax):
            lcs = []
            for c in ax.collections:
                if isinstance(c, mpl.collections.LineCollection):
                    lcs.append(c)
            if len(lcs) > 0:
                return lcs
            raise RuntimeError("No LineCollections found")
        all_num_params = sorted(df['num_params'].unique().tolist())
        df = df[df['network_type']=='quantum']
        quantum_params = sorted(list(df['num_params'].unique()))
        linecollections = get_linecollections(ax)
        color_dict = dict(zip(sorted(df['cutoff'].unique()), colors[6:]))
        for num_param,linecollection in zip(all_num_params, linecollections):
            if num_param == quantum_params[0]:
                df_line_colour = df[df['num_params']==quantum_params.pop(0)].sort_values('val_acc')
                colour_list = [color_dict[i] for i in df_line_colour['cutoff'].values]
                linecollection.set_segments(sorted(linecollection.get_segments(), key=lambda x: x[0][1]))
                linecollection.set_colors(colour_list)
            if not quantum_params:
                break
        return color_dict

    sns.violinplot(df,
                    x='num_params', y='val_acc', hue='network_type', legend=None, 
                    hue_order=['quantum', 'classical'], width=0.8, bw_method=0.1, inner='stick', split=False, cut=0, palette=palette, alpha=1.0,
                    density_norm='count', inner_kws={'lw': 1, 'color': 'grey', 'alpha':0.9}, native_scale=False, ax=ax)


    color_dict = colour_cutoffs(ax, df)
    
    return color_dict

def plot_coloured_photonic_loss_violin(df, ax, cmap='Greys', colorbar_var='photonic_loss', hue='num_qumodes'):
    def colour_cutoffs(ax, df, colour_map):
        def get_linecollections(ax):
            lcs = []
            for c in ax.collections:
                if isinstance(c, mpl.collections.LineCollection):
                    lcs.append(c)
            if len(lcs) > 0:
                return lcs
            raise RuntimeError("No LineCollections found")
        
        linecollections = get_linecollections(ax)
        df.sort_values('val_acc', inplace=True)
        colour_list = [colour_map(i) for i in df[colorbar_var].values]
        for linecollection in linecollections:
            linecollection.set_segments(sorted(linecollection.get_segments(), key=lambda x: x[0][1]))
            linecollection.set_colors(colour_list)

    sns.violinplot(df,
                    x='n_layers', y='val_acc', legend=None, hue=hue,
                    width=0.8, bw_method=0.1, inner='stick', split=False, cut=0, palette=palette, alpha=1.0,
                    density_norm='count', inner_kws={'lw': 1, 'color': 'grey', 'alpha':0.9}, native_scale=False, ax=ax)

    norm = mpl.colors.Normalize(df[colorbar_var].min(), df[colorbar_var].max())
    colour_map = mpl.cm.get_cmap(cmap)
    sm = mpl.cm.ScalarMappable(cmap=colour_map, norm=norm)
    
    
    
    colour_cutoffs(ax, df, lambda x: colour_map(norm(x)))


    
    return norm, sm
 
if __name__ == '__main__': 
    
    # Load the data. Right now I am using the exception line to decide to regenerate the data or not.
    try:
        # raise Exception()
        df_kerr8 = pd.read_pickle('./dataframes/df_kerr8_all_weights_max_amplitudes.pkl', compression='xz')
        df_kerr8.dropna(inplace=True)
    except:
        df_kerr8 = get_maximum_amplitudes(df_kerr8)
        df_kerr8.dropna(inplace=True)
        df_kerr8['max_squeezing_db'] = df_kerr8['max_squeezing'].apply(convert_r_to_db)   
        df_kerr8.to_pickle('./dataframes/df_kerr8_all_weights_max_amplitudes.pkl', compression='xz')

    
    # Take last epoch's data
    df_kerr8['val_acc'] = df_kerr8['val_acc'].apply(lambda x: x[-1])
    df_kerr8['val_loss'] = df_kerr8['val_loss'].apply(lambda x: x[-1])

    max_squeezing = df_kerr8['max_squeezing'].max()

    # Average over different trials 
    groups  = ['num_qumodes', 'cutoff', 'n_layers']
    columns = ['val_acc', 'val_loss', 'max_amplitude', 'max_squeezing', 'max_squeezing_db']
    df_kerr8_mean = df_kerr8.groupby(groups, group_keys=True)[columns].mean()
    df_kerr8_std = df_kerr8.groupby(groups, group_keys=True)[columns].std()

    # Dont average over trials. Take all values. Use if dataframe is already averaged.
    # df_kerr8_mean = df_kerr8[groups + columns]

    # Merge the dataframes
    df_kerr8_mean = pd.merge(df_kerr8_mean, df_kerr8_std, on=groups, suffixes=('', '_std'))

 
    df_kerr8_mean.reset_index(inplace=True)
    df_kerr8_mean = df_kerr8_mean[df_kerr8_mean['num_qumodes'] == 2] 


    df_kerr8_loss = df_kerr8_mean.copy()

    # Calculate loss    
    df_kerr8_loss['eta'] = np.NaN

    # df_kerr8_loss = df_loss_calculation(df_kerr8_loss)

    df_kerr8_loss_full = df_loss_calculation_v2(df_kerr8[groups + columns], max_squeezing=max_squeezing)
    df_kerr8_loss_full['loss_per_layer'] = df_kerr8_loss_full['photonic_loss'] / df_kerr8_loss_full['n_layers']

    # Calculate loss
    df_kerr8_loss_v2 = df_loss_calculation_v2(df_kerr8_mean, max_squeezing=max_squeezing)
    df_kerr8_loss_v2.sort_values('max_squeezing_db', inplace=True)
    df_kerr8_loss_v2['loss_per_layer'] = df_kerr8_loss_v2['photonic_loss'] / df_kerr8_loss_v2['n_layers']
    df_kerr8_loss_v2['loss_per_layer_std'] = df_kerr8_loss_v2['photonic_loss_std'] / df_kerr8_loss_v2['n_layers']

    # # Do the labels for the second axis.
    # max_squeezing_xticklabels = [2.25, 2.75, 3.25, 3.75, 4.25]
    # max_squeezing_xticks = np.interp(max_squeezing_xticklabels, df_kerr8_loss_v2['max_squeezing_db'], df_kerr8_loss_v2['photonic_loss'])
    
    # fig, ax = plt.subplots(1, 2, figsize=(5.5, 4.5), sharey=True)
    # ax1 = ax[0]
    # ax2 = ax1.twiny()
    # ax3 = ax[1]

    # df_hybrid_120 = pd.read_pickle('./dataframes/df_hybrid_120.pkl', compression='xz')

    # color_cutoff_dict = plot_coloured_cutoff_violin(df_hybrid_120, ax3)

    # df_kerr8_loss_120 = df_kerr8_loss_v2[df_kerr8_loss_v2['n_layers'] == 1]

    # for key, val in color_cutoff_dict.items():
    #     label = df_kerr8_loss_120[df_kerr8_loss_120['cutoff'] == key]['photonic_loss'].values[0] 
    #     ax3.plot([], [], color=val, label=f'{label:0.3f} dB', lw=2)

    #     ax3.legend(title='Estimated\nPhotonic Loss', loc='lower right', frameon=False)
    
    # ax3.set_xlabel('')

    # ax3.set_xticklabels(['120 Parameters'], fontsize=12)


    # sns.lineplot(df_kerr8_loss_v2, x='photonic_loss', y='val_acc', markers=True, hue='num_qumodes', style='num_qumodes', palette=palette[0:], legend=None, ax=ax1)
    # # sns.lineplot(df_kerr8_loss_v2, x='max_squeezing_db', y='val_acc', markers=True, hue='num_qumodes', style='num_qumodes', palette=palette[2:], legend=None, ax=ax2)
    # # ax2.cla()
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(max_squeezing_xticks)
    # ax2.set_xticklabels([f'{i:0.2f}' for i in max_squeezing_xticklabels])
    # plt.tight_layout()
    # ax1.set_ylabel('Validation Accuracy')
    # # ax2.set_ylabel('Validation Accuracy')
    # ax1.set_xlabel('Estimated Photonic Loss (dB)')
    # ax2.set_xlabel('Maximum Squeezing (dB)')

    # plt.savefig('./figures/photonic_loss_vs_accuracy.pdf', bbox_inches='tight')
    # plt.savefig('./figures/photonic_loss_vs_accuracy.png', bbox_inches='tight')
    # plt.savefig('./figures/photonic_loss_vs_accuracy.svg', bbox_inches='tight')
    
    # plt.close()

    #Now lets try to plot the distribution of just a handful of points.

    # Select the three points right at the -1 dB cutoff point.
    selected_points_filter = (df_kerr8_loss_v2['photonic_loss'] > -2.2) & (df_kerr8_loss_v2['photonic_loss'] < -1.7)


    df_kerr8_loss_v2_selected = df_kerr8_loss_v2[selected_points_filter]

    # Create filter based on identity of the selected points
    selected_points_filter = df_kerr8_loss_v2[selected_points_filter][['cutoff', 'n_layers', 'num_qumodes']]

    print(df_kerr8_loss_v2_selected)


    # Merge the selected points with the full dataframe with inner strategy
    # Will only keep points with the same cutoff, n_layers and num_qumodes
    # We now have samples only from the three points we selected from the average plot
    df_kerr8_loss_full_selected = df_kerr8_loss_full.merge(selected_points_filter, on=['cutoff', 'n_layers', 'num_qumodes'], how='inner')

    
    sns.histplot(df_kerr8_loss_full_selected, x='photonic_loss', stat='density', hue='n_layers', 
                 element='poly', binwidth=0.3, fill=True, palette=palette[2:], 
                 common_norm=False, legend=None)
    
    # Add lines fot the mean for each n_layers
    for i, n_layers in enumerate(df_kerr8_loss_full_selected['n_layers'].unique()):
        mean = df_kerr8_loss_full_selected[df_kerr8_loss_full_selected['n_layers'] == n_layers]['photonic_loss'].mean()
        std = df_kerr8_loss_full_selected[df_kerr8_loss_full_selected['n_layers'] == n_layers]['photonic_loss'].std()
        plt.errorbar(mean, 1.0-n_layers/30, xerr=std, fmt='x', color=palette[2 + i], markersize=5, capsize=5, capthick=1.0, elinewidth=1.0, label=f'{n_layers} Layers')
        plt.axvline(mean, color=palette[2 + i], linestyle='--', lw=1.0)
    plt.xlabel('Estimated Photonic Loss (dB)')
    
    plt.savefig('./figures/photonic_loss_vs_accuracy_inset1.png', bbox_inches='tight')
    plt.close()

    sns.scatterplot(df_kerr8_loss_full_selected, x='photonic_loss', y='val_acc', hue='n_layers',
                    legend=None, palette=palette[2:])
    
    for i, n_layers in enumerate(df_kerr8_loss_full_selected['n_layers'].unique()):
        mean_loss = df_kerr8_loss_full_selected[df_kerr8_loss_full_selected['n_layers'] == n_layers]['photonic_loss'].mean()
        mean_acc = df_kerr8_loss_full_selected[df_kerr8_loss_full_selected['n_layers'] == n_layers]['val_acc'].mean()
        std_loss = df_kerr8_loss_full_selected[df_kerr8_loss_full_selected['n_layers'] == n_layers]['photonic_loss'].std()
        std_acc = df_kerr8_loss_full_selected[df_kerr8_loss_full_selected['n_layers'] == n_layers]['val_acc'].std()
        plt.errorbar(mean_loss, mean_acc, xerr=std_loss, yerr=std_acc, fmt='x', color=palette[2 + i], markersize=5, capsize=5, capthick=1.0, elinewidth=1.0, label=f'{n_layers} Layers')
        plt.axvline(mean_loss, color=palette[2 + i], linestyle='--', lw=1.0)
    
    plt.xlabel('Estimated Photonic Loss (dB)')
    plt.ylabel('Validation Accuracy')
    plt.savefig('./figures/photonic_loss_vs_accuracy_inset2.png', bbox_inches='tight')
    plt.close()

 
    max_squeezing_xticklabels = [2.25, 3.25, 4.25]
    max_squeezing_xticks = np.interp(max_squeezing_xticklabels, df_kerr8_loss_v2['max_squeezing_db'], df_kerr8_loss_v2['photonic_loss'])
    
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 4.5), sharey=True)
    ax1 = axs[1]
    ax2 = axs[0]
    ax3 = ax1.twiny()
    cbax = fig.add_axes([0.11, 0.90, 0.42, 0.03])
    df_kerr8_loss_full = df_kerr8_loss_full[df_kerr8_loss_full['num_qumodes'] == 2]
    norm, sm = plot_coloured_photonic_loss_violin(df_kerr8_loss_full, ax2)
    plt.colorbar(sm, cax=cbax, label='Estimated Photonic Loss (dB)', orientation='horizontal')
    cbax.xaxis.set_ticks_position('top')
    cbax.xaxis.set_label_position('top')
    ax2.set_xlabel('Number of Layers')
    # ax1.set_xticklabels(["Two Qumodes\nN Layers"], fontsize=12)

    # sns.lineplot(df_kerr8_loss_v2, x='photonic_loss', y='val_acc', markers=True, hue='num_qumodes', style='num_qumodes', palette=palette[0:], legend=None, ax=ax1)
    
    x_vals = df_kerr8_loss_v2['photonic_loss']
    y_vals = df_kerr8_loss_v2['val_acc']
    x_err = df_kerr8_loss_v2['photonic_loss_std']
    y_err = df_kerr8_loss_v2['val_acc_std']

    sns.scatterplot(df_kerr8_loss_v2_selected, ax=ax3, x='photonic_loss', y='val_acc', hue='n_layers', palette=palette[2:], legend=None)
    ax1.errorbar(x_vals, y_vals, xerr=x_err, fmt='o', color=palette[0], markersize=3, capsize=2, capthick=0.5, elinewidth=0.5, label='Two Qumodes\nN Layers')

    
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(max_squeezing_xticks)
    ax3.set_xticklabels([f'{i:0.2f}' for i in max_squeezing_xticklabels])
    ax3.set_xlabel('Maximum Squeezing (dB)')
    
    
    ax2.set_ylabel('Validation Accuracy')
    ax1.set_xlabel('Estimated Photonic Loss (dB)')
    
    plt.tight_layout()

    plt.savefig('./figures/photonic_loss_vs_accuracy_v2.pdf', bbox_inches='tight')
    plt.savefig('./figures/photonic_loss_vs_accuracy_v2.png', bbox_inches='tight')
    plt.savefig('./figures/photonic_loss_vs_accuracy_v2.svg', bbox_inches='tight')
    
    plt.close()
   
    print('Correlation') 
    print(df_kerr8_loss_full[['val_acc', 'cutoff', 'n_layers', 'num_qumodes', 'max_squeezing']].corr())

    
    # Plot the loss per layer
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    sns.lineplot(df_kerr8_loss_v2, x='photonic_loss', y='loss_per_layer', markers=True, hue='n_layers', palette=palette[0:], ax=ax)

    plt.savefig('./figures/loss_per_layer.png', bbox_inches='tight')
    plt.close()

    # Recreate loss plot with loss per layer
    max_squeezing_xticklabels = [2.25, 3.25, 4.25]
    max_squeezing_xticks = np.interp(max_squeezing_xticklabels, df_kerr8_loss_v2['max_squeezing_db'], df_kerr8_loss_v2['loss_per_layer'])
    
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 4.5), sharey=True)
    ax1 = axs[1]
    ax2 = axs[0]
    cbax = fig.add_axes([0.11, 0.95, 0.42, 0.03])
    df_kerr8_loss_full = df_kerr8_loss_full[df_kerr8_loss_full['num_qumodes'] == 2]
    norm, sm = plot_coloured_photonic_loss_violin(df_kerr8_loss_full, ax2, colorbar_var='loss_per_layer')
    plt.colorbar(sm, cax=cbax, label='Estimated Photonic Loss (dB/layer)', orientation='horizontal')
    cbax.xaxis.set_ticks_position('top')
    cbax.xaxis.set_label_position('top')
    ax2.set_xlabel('Number of Layers')
    # ax1.set_xticklabels(["Two Qumodes\nN Layers"], fontsize=12)

    # sns.lineplot(df_kerr8_loss_v2, x='photonic_loss', y='val_acc', markers=True, hue='num_qumodes', style='num_qumodes', palette=palette[0:], legend=None, ax=ax1)
    
    x_vals = df_kerr8_loss_v2['loss_per_layer']
    y_vals = df_kerr8_loss_v2['val_acc']
    x_err = df_kerr8_loss_v2['loss_per_layer_std']
    y_err = df_kerr8_loss_v2['val_acc_std']

    ax1.errorbar(x_vals, y_vals, xerr=x_err, fmt='o', color=palette[0], markersize=3, capsize=2, capthick=0.5, elinewidth=0.5, label='Two Qumodes\nN Layers', zorder=0)
    sns.scatterplot(df_kerr8_loss_v2_selected, ax=ax1, x='loss_per_layer', y='val_acc', hue='n_layers', palette=palette[2:], legend=None, zorder=1)

    
    
    
    ax2.set_ylabel('Validation Accuracy')
    ax1.set_xlabel('Estimated Photonic Loss (dB/layer)')
    
    plt.tight_layout()

    plt.savefig('./figures/loss_per_layer_vs_accuracy_v2.pdf', bbox_inches='tight')
    plt.savefig('./figures/loss_per_layer_vs_accuracy_v2.png', bbox_inches='tight')
    plt.savefig('./figures/loss_per_layer_vs_accuracy_v2.svg', bbox_inches='tight')
    
    plt.close()

    
    # Recreate loss plot with different n_layers
    max_squeezing_xticklabels = np.array([5., 4., 3., 2.])
    max_squeezing_xticklabels_r = convert_db_to_r(max_squeezing_xticklabels)
    max_squeezing_xticks = 10*np.log10(calculate_loss_from_variance(max_squeezing, max_squeezing_xticklabels_r))
    
    
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 4.5), sharey=True)
    ax1 = axs[1]
    ax2 = axs[0]
    ax3 = ax1.twiny()
    cbax = fig.add_axes([0.11, 0.90, 0.42, 0.03])
    df_kerr8_loss_full = df_kerr8_loss_full[df_kerr8_loss_full['num_qumodes'] == 2]
    norm, sm = plot_coloured_photonic_loss_violin(df_kerr8_loss_full, ax2, colorbar_var='photonic_loss', hue='n_layers')
    plt.colorbar(sm, cax=cbax, label='Estimated Photonic Loss (dB)', orientation='horizontal')
    cbax.xaxis.set_ticks_position('top')
    cbax.xaxis.set_label_position('top')
    ax2.set_xlabel('Number of Layers')
    # ax1.set_xticklabels(["Two Qumodes\nN Layers"], fontsize=12)

    # sns.lineplot(df_kerr8_loss_v2, x='photonic_loss', y='val_acc', markers=True, hue='num_qumodes', style='num_qumodes', palette=palette[0:], legend=None, ax=ax1)
    for i, n_layers in enumerate([1, 2, 3, 4, 5]):
        x_vals = df_kerr8_loss_v2[df_kerr8_loss_v2['n_layers'] == n_layers]['photonic_loss']
        y_vals = df_kerr8_loss_v2[df_kerr8_loss_v2['n_layers'] == n_layers]['val_acc']
        x_err = df_kerr8_loss_v2[df_kerr8_loss_v2['n_layers'] == n_layers]['photonic_loss_std']
        y_err = df_kerr8_loss_v2[df_kerr8_loss_v2['n_layers'] == n_layers]['val_acc_std']
        ax1.errorbar(x_vals, y_vals, xerr=x_err, fmt='o', color=palette[i], markersize=3, capsize=2, capthick=0.5, elinewidth=0.5, label=f'{n_layers} Layers', zorder=0)
    
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(max_squeezing_xticks)
    ax3.set_xticklabels([f'{i:0.2f}' for i in max_squeezing_xticklabels])
    ax3.set_xlabel('Maximum Squeezing (dB)')
 
    y_aid = np.linspace(-5, 0, 100)
    x_vals = df_kerr8_loss_v2['photonic_loss']
    y_vals = df_kerr8_loss_v2['val_acc']
    
    ax2.set_ylabel('Validation Accuracy')
    ax1.set_xlabel('Estimated Photonic Loss (dB)')
    
    plt.tight_layout()

    plt.savefig('./figures/photonic_loss_vs_accuracy_v3.pdf', bbox_inches='tight')
    plt.savefig('./figures/photonic_loss_vs_accuracy_v3.png', bbox_inches='tight')
    plt.savefig('./figures/photonic_loss_vs_accuracy_v3.svg', bbox_inches='tight')
    
    plt.close()

    # Calculate the reduction of accuracy for all accuracy values due to greater photonic loss values
    df_kerr8_loss_v2_sorted = df_kerr8_loss_v2.sort_values(by=['val_acc'])
    df_kerr8_loss_v2_sorted['accuracy_reduction'] = df_kerr8_loss_v2_sorted['val_acc'].transform(lambda x: x.max() - x)

    # Create a DataFrame to present the data in a table
    accuracy_table = df_kerr8_loss_v2_sorted[['val_acc', 'n_layers', 'photonic_loss', 'accuracy_reduction', 'max_squeezing', 'max_squeezing_db']].copy()
    # Print the table
    print('Accuracy Reduction Table')
    print(accuracy_table) 

    from sklearn.linear_model import LinearRegression

    X = df_kerr8_loss_v2['photonic_loss'].values.reshape(-1, 1)
    Y = df_kerr8_loss_v2['val_acc'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    print(f'Coefficient of determination: {reg.score(X, Y)}')
    print(f'Coefficient: {reg.coef_}')

    print('Performance at -8 dB of loss:', reg.predict(np.array([[-8]]))[0])

    # Calculate the required initial squeezing to achieve a certain measured squeezing with a given photonic loss

    # The measured squeezing is 3.59 dB
    measured_squeezing = 3.59
    measured_squeezing_r = convert_db_to_r(measured_squeezing)
    print(measured_squeezing_r)
    # The photonic loss is 1.8 dB
    photonic_loss = -1.8
    photonic_loss_eta = 10**(photonic_loss/10)
    print(photonic_loss_eta)
    # The initial squeezing
    initial_squeezing = calculate_initial_variance_from_loss(measured_squeezing_r, photonic_loss_eta)
    print(f'Initial squeezing to achieve {measured_squeezing} dB with a photonic loss of {photonic_loss} dB: {convert_to_db(initial_squeezing)} dB')
    
    