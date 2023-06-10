from keras import Sequential
from keras import layers
from keras import optimizers
from keras_tuner.tuners import RandomSearch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from keras import callbacks


def build_model_nn(hp):
    model = Sequential()
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_tuned_nn(x_train, y_train, use_callback = True):
    if use_callback:
        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner = RandomSearch(build_model_nn, objective='val_accuracy',
                         max_trials=5, overwrite=True, directory='./project', callbacks = [stop_early])
    else: 
        tuner = RandomSearch(build_model_nn, objective='val_accuracy',
                         max_trials=5, overwrite=True, directory='./project')
    
    # Perform hyperparameter search
    tuner.search(x_train, y_train, epochs=100, validation_split=0.15)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    return model, best_hps

param_grid_rfc = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt', None]
}


param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}


def build_tuned_rfc(x_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = param_grid_rfc
    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, n_jobs=8,
                                   random_state=42)
    rf_random.fit(x_train, y_train)
    best_params = rf_random.best_params_
    return rf_random, best_params


def build_tuned_svc(x_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = param_grid_svc
    svc = svm.SVC()

    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=3, n_jobs=8)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    return grid_search, best_params


def create_nn(input_shape, compile=True):
    # create a model
    model = Sequential([
        layers.Input(shape = input_shape,),
    layers.Dense(80, activation='relu'),
    layers.Dense(60, activation='relu'),
    layers.Dense(40, activation='relu'),
    layers.Dense(1, activation='sigmoid')])
    # compile a model
    if compile:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

