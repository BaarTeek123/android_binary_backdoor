from keras import Sequential
from keras import layers
from keras import optimizers
from keras_tuner.tuners import RandomSearch


def build_model_nn(hp):
    model = Sequential()
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_optimal_nn(x_train, y_train):
    tuner = RandomSearch(build_model_nn, objective='val_accuracy',
                         max_trials=5, overwrite=True, directory='./project')

    # Perform hyperparameter search
    tuner.search(x_train, y_train, epochs=10, validation_split=0.1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    return model
