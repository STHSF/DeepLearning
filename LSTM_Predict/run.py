import time
import lstm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

HIDDEN_SIZE = 30
NUM_LAYERS = 2
BATCH_SIZE = 32
TRAINING_STEPS = 100


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# Main Run Thread
if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 1
    timesteps = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', timesteps, normalise_window=True)

    print(np.shape(X_train))
    print('X_train.shape\n', np.shape(X_train))
    # print('X_train\n', X_train)
    print('y_train.shape\n', np.shape(np.array(y_train, dtype=np.float32)))
    print('X_test.shape\n', np.shape(X_test))
    print('y_test.shape\n', np.shape(y_test))
    print('y_test:\n', y_test[0])

    print('> Data Loaded. Compiling...')

    # layers = [input_dim, input_length, hidden_units, output_dim]
    model = lstm.build_model([1, 50, 100, 1])
    #
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=epochs,
        validation_split=0.05)

    predictions = lstm.predict_sequences_multiple(model, X_test, timesteps, 50)
    # predictions = lstm.predict_sequence_full(model, X_test, timesteps)
    # predictions = lstm.predict_point_by_point(model, X_test)

    print('Training duration (s) : ', time.time() - global_start_time)
    plot_results_multiple(predictions, y_test, 50)
    # plot_results(predictions, y_test)
