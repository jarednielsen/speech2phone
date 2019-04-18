"""Optimize some hyperparameters.

Kyle Roth. 2019-04-11.
"""


from time import time
import os
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def test_model(X_train, y_train, X_val, y_val, widths, lr, decay, momentum, epochs, batch_size, ret_model=False):
    """Train the model and return results.

    Args:
        X_train (np.ndarray).
        y_train (np.ndarray).
        X_val (np.ndarray).
        y_val (np.ndarray).
        widths (list(int)): widths for each hidden layer. There will be len(widths) hidden layers.
        lr (float): learning rate.
        decay (float): learning rate decay; lr *= (1. / (1. + decay * iterations)).
        momentum (float).
        epochs (int).
        batch_size (int).
        ret_model (bool): whether to return the model when finished.
    Returns:
        (float): time spent training.
        (float): accuracy on the validation set.
    Saves:
        (training/{}.png): PNG file with the loss and accuracy during training.
    """
    # build model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    for width in widths:
        model.add(keras.layers.Dense(width, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=lr,
                                                 decay=decay,
                                                 momentum=momentum,
                                                 nesterov=False),
                  metrics=['accuracy'])
    
    # train model and time it
    start = time()
    history = model.fit(X_train, y_train, 
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val))
    train_time = time() - start
    
    # evaluate
    plt.figure(figsize=(8,5))
    plt.plot(history.history['acc'], label='training accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.1)
    plt.ylim(0, 1)
    plt.title('Widths ({}), LR ({}), Decay ({}), Momentum ({}), Epochs ({}), Batch ({})'.format(widths, lr, decay, momentum, epochs, batch_size))
    os.makedirs('training', exist_ok=True)
    plt.savefig('training/{}|{}|{}|{}|{}|{}.png'.format(widths, lr, decay, momentum, epochs, batch_size))
    
    if ret_model:
        return train_time, model.evaluate(X_val, y_val)[1], model
    return train_time, model.evaluate(X_val, y_val)[1]


def main():
    # import data
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # rescale to be in the interval [0,1], this should learn better, since 
    # initialization happens with standard normal, or similarly small values.
    X_train_full = X_train_full / 255
    X_test = X_test / 255
    
    # make a validation split beyond the basic test/train
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                      test_size=0.2, 
                                                      random_state=42)
    
    # hyperparameter space
    widthses = [
        [300, 100],
        [100, 100, 100],
        [200, 200],
        [300, 300]
    ]
    lrs = [0.01, 0.002]
    decays = [1e-6, 0.0, 1e-5]
    momenta = [0.8, 0.0]
    epochses = [10, 20]
    batch_sizes = [32, 64]
    # widthses = [
    #     [300, 100],
    #     [100, 100]
    # ]
    # lrs = [0.01]
    # decays = [1e-6]
    # momenta = [0.8]
    # epochses = [10]
    # batch_sizes = [32]
    
    results = {}
    # try all combinations
    for widths in widthses:
        for lr in lrs:
            for decay in decays:
                for momentum in momenta:
                    for epochs in epochses:
                        for batch_size in batch_sizes:
                            train_time, acc = test_model(X_train, y_train, X_val, y_val, widths, lr, decay, momentum, epochs, batch_size)
                            results[acc] = [train_time, [widths, lr, decay, momentum, epochs, batch_size]]
    
    with open('training/results.txt', 'w+') as outfile:
        for acc in results:
            outfile.write('{}: {}\n'.format(acc, str(results[acc])))


if __name__ == '__main__':
    main()
