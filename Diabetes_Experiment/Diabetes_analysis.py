import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"Set Matplotlib Defaults"
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

def get_metrics(filename):
    with open(filename) as json_file:
        data = json.load(json_file)

    metrics = {}
    metrics['Training Accuracy'] = data['accuracy']["values"]
    metrics['Training Loss'] = data['loss']["values"]
    metrics['Testing Accuracy'] = data['val_accuracy']["values"]
    metrics['Testing Loss'] = data['val_loss']["values"]
    metrics['Norm'] = data['normalization']["values"]

    return metrics

def get_accuracy_data(filepath, num_files):


    for file_num in range(1,num_files+1):
        with open(filepath+'/'+str(file_num)+'config.json') as json_file:
            config = json.load(json_file)


def plot_example(filepath, num_files, initial_weight_amplitudes, initial_input_amplitude, loss_coefficient):

    metrics = {}
    for file_num in range(1,num_files+1):
        with open(filepath+'/'+str(file_num)+'/config.json') as json_file:
            config = json.load(json_file)

        if(config["initial_weight_amplitudes"]==initial_weight_amplitudes
            and config["initial_input_amplitude"] == initial_input_amplitude
            and config["loss_coefficient"]==loss_coefficient):

            if(config["cutoff_management"]=="Loss"):
                metrics['loss'] = get_metrics(filepath+'/'+str(file_num)+'/metrics.json')
            elif(config["cutoff_management"]=="L1"):
                metrics['L1'] = get_metrics(filepath + '/' + str(file_num) + '/metrics.json')
            elif (config["cutoff_management"] == "L2"):
                metrics['L2'] = get_metrics(filepath + '/' + str(file_num) + '/metrics.json')
            elif (config["cutoff_management"] == None):
                metrics['None'] = get_metrics(filepath + '/' + str(file_num) + '/metrics.json')
            else:
                print("no valid cutoff management strategy found.")

    method_names = ['loss','L1','L2','None']
    metric_names = ['Training Accuracy', 'Training Loss','Testing Accuracy','Testing Loss','State Size']

    n_epochs = len(metrics['loss']['Training Accuracy'])
    epochs = np.arange(1, n_epochs + 1)

    for metric_name in metric_names:

        fig, axes = plt.subplots(1, figsize=(3, 3))
        colors = ['r', 'b', 'g', 'c']

        for idx, method_name in enumerate(method_names):
            axes.plot(epochs, metrics[method_name][metric_name], colors[idx], linewidth=0.4)

        axes.set_xlim(0)
        axes.grid(True, linestyle=':')
        axes.set_xlabel("Epoch")
        axes.set_ylabel(metric_name)
        fig.legend(labels=method_names, ncol=2, loc="lower left", bbox_to_anchor=(0.32,0.3), borderaxespad=0.)
        fig.tight_layout(pad=0.3)

        plt.savefig('test' + '.pdf',
                    format='pdf',
                    dpi=1200,
                    bbox_inches='tight')

        plt.show()


def plot_final_accuracy_loss(filepath, num_files, initial_weight_amplitudes, initial_input_amplitude, loss_coeffs, save_plot=False):
    method_names = ['Loss', 'L1', 'L2', None]
    final_accuracies = {}
    convergence_speed = {}
    for method_name in method_names:
        final_accuracies[method_name]=[]
        convergence_speed[method_name] = []

    for loss_coeff in loss_coeffs:
        for file_num in range(1,num_files+1):
            with open(filepath+'/'+str(file_num)+'/config.json') as json_file:
                config = json.load(json_file)

            if(config["initial_weight_amplitudes"]==initial_weight_amplitudes
                and config["initial_input_amplitude"] == initial_input_amplitude
                and config["loss_coefficient"] == loss_coeff):

                metrics = get_metrics(filepath+'/'+str(file_num)+'/metrics.json')

                for method_name in method_names:
                    if(config["cutoff_management"]==method_name):
                        final_accuracies[method_name].append(metrics["Testing Accuracy"][-1])

                        epoch = 0
                        while(metrics["Testing Accuracy"][epoch]<0.85 and epoch<len(metrics["Testing Accuracy"])):
                            epoch += 1
                        convergence_speed[method_name].append(epoch)

    fig, axes = plt.subplots(1, figsize=(3, 3))
    colors = ['r', 'b', 'g', 'c']

    for idx, method_name in enumerate(method_names):
        axes.plot(loss_coeffs, final_accuracies[method_name], colors[idx], linewidth=0.9)

    axes.set_xlim(0)
    axes.grid(True, linestyle=':')
    axes.set_xlabel("Loss Coefficient")
    axes.set_ylabel("Final Testing Accuracy")
    fig.legend(labels=method_names, ncol=2, loc="lower left", bbox_to_anchor=(0.32,1), borderaxespad=0.)
    fig.tight_layout(pad=0.3)
    plt.show()

def plot_metrics(filepath, num_files, cutoff_dimension, encoding_strategy, save_plot=False):

    # Loop through files until desired configuration found
    for file_num in range(1,num_files+1):
        with open(filepath+'/'+str(file_num)+'/config.json') as json_file:
            config = json.load(json_file)

        if(config["cutoff_dimension"]==cutoff_dimension
            and config["encoding_strategy"] == encoding_strategy):
            metrics = get_metrics(filepath+'/'+str(file_num)+'/metrics.json')

            n_epochs = len(metrics['Training Accuracy'])
            epochs = np.arange(1, n_epochs + 1)

            # Plot accuracy and normalization
            fig, axes = plt.subplots(1, figsize=(3, 3))
            colors = ['r', 'b', 'g']
            metric_names1 = ['Training Accuracy', 'Testing Accuracy', 'State Size']
            for i in range(3):

                axes.plot(epochs, metrics[metric_names1[i]], colors[i])

            axes.set_xlim(0)
            axes.grid(True, linestyle=':')
            axes.set_xlabel("Epoch")
            axes.set_ylabel("Accuracy & Normalization")
            fig.legend(labels=metric_names1, ncol=2, loc="lower left", bbox_to_anchor=(0.1, 0.5), borderaxespad=0.)
            fig.tight_layout(pad=0.3)

            if save_plot:
                plt.savefig(filepath+'/'+str(file_num)+'/acc_norm' + '.pdf',
                            format='pdf',
                            dpi=1200,
                            bbox_inches='tight')

            plt.show()

            # Plot loss
            fig, axes = plt.subplots(1, figsize=(3, 3))
            colors = ['r', 'b']
            metric_names1 = ['Training Loss', 'Testing Loss']
            for i in range(2):
                axes.plot(epochs, metrics[metric_names1[i]], colors[i])

            axes.set_xlim(0)
            axes.grid(True, linestyle=':')
            axes.set_xlabel("Epoch")
            axes.set_ylabel("Loss")
            fig.legend(labels=metric_names1, ncol=2, loc="lower left", bbox_to_anchor=(0.1, 0.5), borderaxespad=0.)
            fig.tight_layout(pad=0.3)

            if save_plot:
                plt.savefig(filepath+'/'+str(file_num)+'/loss' + '.pdf',
                            format='pdf',
                            dpi=1200,
                            bbox_inches='tight')

            plt.show()

def plot_convergence_speed(filepath, num_files, initial_weight_amplitudes, initial_input_amplitudes, loss_coefficient, cutoff_dimension, strategy, save_plot=False):

    convergence_grid = np.zeros((len(initial_weight_amplitudes), len(initial_input_amplitudes)))
    final_acc_grid = np.zeros((len(initial_weight_amplitudes), len(initial_input_amplitudes)))
    initial_norm = np.zeros((len(initial_weight_amplitudes), len(initial_input_amplitudes)))
    final_norm = np.zeros((len(initial_weight_amplitudes), len(initial_input_amplitudes)))
    delta_norm = np.zeros((len(initial_weight_amplitudes), len(initial_input_amplitudes)))

    for file_num in range(1,num_files+1):
        with open(filepath+'/'+str(file_num)+'/config.json') as json_file:
            config = json.load(json_file)

        if(config["initial_weight_amplitudes"] in initial_weight_amplitudes
            and config["initial_input_amplitude"] in initial_input_amplitudes
            and config["loss_coefficient"]==loss_coefficient
            and config["cutoff_management"]==strategy
            and config["cutoff_dimension"]==cutoff_dimension):

            metrics = get_metrics(filepath+'/'+str(file_num)+'/metrics.json')
            n_epochs = len(metrics['Training Accuracy'])
            accuracy = metrics['Testing Accuracy']

            # Find epoch threshold
            epoch = 0
            while(accuracy[epoch]<0.89 and epoch<(n_epochs-1)):
                epoch += 1

            epoch += 1

            # Build meshgrids
            idx_1 = initial_weight_amplitudes.index(config["initial_weight_amplitudes"])
            idx_2 = initial_input_amplitudes.index(config["initial_input_amplitude"])
            convergence_grid[idx_1, idx_2] = epoch
            final_acc_grid[idx_1, idx_2] = metrics['Testing Accuracy'][-1]
            initial_norm[idx_1, idx_2] = metrics['State Size'][0]
            final_norm[idx_1, idx_2] = metrics['State Size'][-1]
            delta_norm[idx_1, idx_2] = metrics['State Size'][-1] - metrics['State Size'][0]

    def density_plot(grid_vals, param_name, save_name):
        fig, axes = plt.subplots(1, figsize=(3.2, 3.2))
        cs = axes.contourf(initial_input_amplitudes, initial_weight_amplitudes, grid_vals, cmap='Blues')
        axes.set_xlabel("Initial Input Value")
        axes.set_ylabel("Initial Weight Amplitude")

        norm = mpl.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ticks=cs.levels)
        cbar.ax.set_ylabel(param_name)
        cbar.ax.tick_params(size=0)
        fig.tight_layout(pad=0.3)

        if save_plot:
            plt.savefig(filepath + '/' + str(file_num) + '/' + save_name + '.pdf',
                        format='pdf',
                        dpi=300,
                        bbox_inches='tight')

        plt.show()

    print(file_num)

    density_plot(convergence_grid, "Iterations to 90% Testing Accuracy", "convergence")
    density_plot(final_acc_grid, "Final Accuracy", "final_accuracy")
    density_plot(initial_norm, "Initial State Size", "initial_norm")
    density_plot(final_norm, "Final State Size", "final_norm")
    density_plot(delta_norm, "Change in State Size", "delta_norm")

def compare_metric(filepath, num_files, metric, cutoff_dimension, encoding_strategies, save_plot=False):

    fig, axes = plt.subplots(1, figsize=(3, 3))
    colors = ['r', 'b', 'g']

    # Loop through files until desired configuration found
    for file_num in range(1,num_files+1):
        with open(filepath+'/'+str(file_num)+'/config.json') as json_file:
            config = json.load(json_file)

        if(config["cutoff_dimension"] == cutoff_dimension
            and config["encoding_strategy"] in encoding_strategies):
            metrics = get_metrics(filepath+'/'+str(file_num)+'/metrics.json')

            n_epochs = len(metrics['Training Accuracy'])
            epochs = np.arange(1, n_epochs + 1)

            axes.plot(epochs, metrics[metric], colors[file_num-1])

    axes.set_xlim(0)
    axes.grid(True, linestyle=':')
    axes.set_xlabel("Epoch")
    axes.set_ylabel(metric)
    fig.legend(labels=encoding_strategies, ncol=2, loc="lower left", bbox_to_anchor=(0.01, 0.15), borderaxespad=0.)
    fig.tight_layout(pad=0.3)

    if save_plot:
        plt.savefig(filepath+'/'+str(file_num)+'/acc_norm' + '.pdf',
                    format='pdf',
                    dpi=1200,
                    bbox_inches='tight')

    plt.show()

#%%
plot_metrics(filepath='./Diabetes_Experiment/Experiment_Data1',
             num_files=3,
             cutoff_dimension = 15,
             encoding_strategy="Sigmoid",
             save_plot=True)

#%%

compare_metric(filepath='./Diabetes_Experiment/Experiment_Data1',
             num_files=3,
             metric = "Training Accuracy",
             cutoff_dimension = 15,
             encoding_strategies=["None", "Sigmoid_BatchNorm", "Sigmoid"],
             save_plot=True)


