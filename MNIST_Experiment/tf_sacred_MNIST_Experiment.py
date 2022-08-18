#%% Imports

# Use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%
# TensorFlow
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# NumPy

# Custom Quantum Layers
from TensorFlow_Archive.tf_quantum_layers import *

# Sacred Package for Experiment Management
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

# Timing
import time

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

#%% Setup Experiment
ex = Experiment('MNIST_TEST')
ex.observers.append(FileStorageObserver(''))
ex.captured_out_filter = apply_backspaces_and_linefeeds

#%% Experiment Parameters
@ex.config
def confnet_config():
    num_epochs = 3
    num_neurons = 20
    activation_type = 'softmax'

#%% Logs
@ex.capture
def log_performance(_run, logs, epoch, time):
    _run.log_scalar("loss", float(logs.get('loss')), epoch)
    _run.log_scalar("accuracy", float(logs.get('accuracy')), epoch)
    _run.log_scalar("val_loss", float(logs.get('val_loss')), epoch)
    _run.log_scalar("val_accuracy", float(logs.get('val_accuracy')), epoch)
    _run.log_scalar("normalization", float(logs.get('val_accuracy')), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)
    _run.log_scalar("time", float(time), epoch)

#%% Main Experiment
#@ex.automain
def define_and_train(num_epochs, num_neurons, activation_type):

    # Metric Logging Callback
    class LogPerformance(Callback):
        def __init__(self):
            super(LogPerformance, self).__init__()

        def on_epoch_begin(self, epoch, logs={}):
            self.start_time=time.time()

        def on_epoch_end(self, epoch, logs={}):
            stop_time=time.time()
            duration = stop_time-self.start_time
            #log_performance(logs=logs, epoch=epoch, time=duration)

    # Timing Estimator
    class TimingEstimator(Callback):
        def __init__(self, num_epochs, num_batches):
            super(TimingEstimator, self).__init__()
            self.num_epochs = num_epochs
            self.num_batches = num_batches

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_batch_begin(self, batch, logs=None):
            if(batch==1):
                self.start_time = time.time()

            if(batch==6):
                stop_time = time.time()
                batch_time = ((stop_time - self.start_time)/5)/3600 # In hours
                epoch_time_estimate = batch_time*self.num_batches
                full_time_estimate = epoch_time_estimate*(num_epochs-self.epoch)
                print("\nEpoch Estimated Time: %.2f hours, Remaining Epochs Estimated Time: %.2f hours"%(epoch_time_estimate, full_time_estimate))

    # Load Data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    #x_train = np.expand_dims(x_train, -1)  # DELETE LATER
    #x_test = np.expand_dims(x_test, -1)  # DELETE LATER

    # train_filter = np.where((y_train == 4) | (y_train == 9) | (y_train == 8))
    # test_filter = np.where((y_test == 4) | (y_test == 9) | (y_test == 8))

    # x_train, y_train = x_train[train_filter], y_train[train_filter]
    # x_test, y_test = x_test[test_filter], y_test[test_filter]

    # y_train[np.where(y_train == 4)] = 0
    # y_train[np.where(y_train == 9)] = 1
    # y_train[np.where(y_train == 8)] = 2
    # y_test[np.where(y_test == 4)] = 0
    # y_test[np.where(y_test == 9)] = 1
    # y_test[np.where(y_test == 8)] = 2

    num_train_samples = 100
    num_test_samples = 10

    x_train, y_train = x_train[:num_train_samples], y_train[:num_train_samples]
    x_test, y_test = x_test[:num_test_samples], y_test[:num_test_samples]

    x_train = tf.convert_to_tensor(x_train, dtype='float32')
    x_test = tf.convert_to_tensor(x_test, dtype='float32')

    # Build Model
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()

            self.main = tf.keras.models.Sequential([
                # layers.Conv2D(600, kernel_size=(3, 3), activation="relu"),  # DELETE LATER
                # layers.MaxPooling2D(pool_size=(2, 2)),  # DELETE LATER
                # layers.Conv2D(200, kernel_size=(3, 3), activation="relu"),  # DELETE LATER
                # layers.MaxPooling2D(pool_size=(2, 2)),  # DELETE LATER
                # layers.Flatten(),  # DELETE LATER
                # tf.keras.layers.Dense(64, activation='relu'),
                # tf.keras.layers.Dense(num_neurons, activation=activation_type),
                # tf.keras.layers.Dense(10, activation='softmax')

                tf.keras.layers.Flatten(input_shape=(28,28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(4, activation='relu'),
                QuantumLayer(4, "cv_neural_net", n_qubits=4, n_blocks=1, cutoff_dim=8),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

        def call(self, inputs):
            x = self.main(inputs)
            return x

    # Compile Model
    model = Net()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(0.01),
                  metrics=['accuracy']
                  )

    batch_size = 10
    num_batches = len(x_train)/batch_size
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              initial_epoch=0,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[LogPerformance(), TimingEstimator(num_epochs, num_batches)]
              )


define_and_train(1, 0, 0)
