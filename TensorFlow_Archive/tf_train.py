#%% Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os.path

#%% Train
def train(num_epochs, batch_size, model, optimizer, loss_function, x_train, y_train, model_path, metrics_path,
          continue_training=False):
    if continue_training:
        model.load_weights(model_path)
        metrics = np.load(metrics_path)
        start_epoch = metrics['epochs'][-1]+1
        epochs = metrics['epochs']
        accuracy = metrics['training_accuracy']
        loss = metrics['training_loss']
    else:
        if os.path.isfile(model_path) or os.path.isfile(metrics_path):
            raise Exception("File already exists. Please write to a new file.")
        start_epoch = 0
        epochs = np.array([]).astype(int)
        accuracy = np.array([])
        loss = np.array([])

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy']
                  )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=False
    )

    class Metrics(tf.keras.callbacks.Callback):
        def __init__(self, accuracy, loss, epochs):
            super(Metrics, self).__init__()
            self.accuracy = accuracy
            self.loss = loss
            self.epochs = epochs
        def on_epoch_end(self, epoch, logs=None):
            self.accuracy = np.append(self.accuracy, logs["accuracy"])
            self.loss = np.append(self.loss, logs["loss"])
            self.epochs = np.append(self.epochs, int(epoch))
            np.savez(metrics_path, training_accuracy=self.accuracy, training_loss=self.loss, epochs=self.epochs)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, initial_epoch = start_epoch, callbacks=[cp_callback, Metrics(accuracy, loss, epochs)])

#%% Test
def test(model, x_test, y_test, batch_size, metrics_path):
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    metrics = np.load(metrics_path)
    epochs = metrics['epochs']
    accuracy = metrics['training_accuracy']
    loss = metrics['training_loss']
    np.savez(metrics_path, training_accuracy=accuracy, training_loss=loss, epochs=epochs, testing_accuracy=results[1], testing_loss=results[0])
