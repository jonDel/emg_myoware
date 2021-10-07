from types import FunctionType
from typing import List
from bisect import bisect_left
import sys
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
import plotly.express as px
from numpy import ndarray
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Lambda, LSTM,\
    Dropout, Reshape, TimeDistributed, Convolution2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf


tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
HAS_GPU = bool(tf.test.gpu_device_name())


def sync_classes(measures_file: str, classes_file: str, subsamp_rate: int =1) -> DataFrame:
    m_df = pd.read_csv(measures_file)[::subsamp_rate]
    c_df = pd.read_csv(classes_file)
    m_df["class"] = [0]*len(m_df)
    curr_class = 0
    for index, row in m_df.iterrows():
        idx = bisect_left(c_df['time'].values, row['time'])
        if idx < len(c_df):
            m_class = c_df["class"].iloc[idx-1]
            c_time = c_df["time"].iloc[idx-1]
            m_df.at[index, "class"] = m_class
    return m_df


def combine_datasets(dataframes: List[DataFrame]):
    time_offset = 0
    cp_dataframes = [df.copy() for df in dataframes]
    for dataframe in cp_dataframes:
        dataframe["time"] = dataframe["time"] + time_offset
        time_offset = dataframe["time"].iloc[-1]
    return pd.concat(cp_dataframes)


def train(dataframe: DataFrame, clf, split_size: float=0.3,
          target_names: list = [],
          plot_confusion_matrix: bool=True) -> list:
    X = dataframe["semg"].to_numpy().reshape(-1, 1)
    y = dataframe["class"]
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if plot_confusion_matrix:
        plot_confusion_matrix(clf, X_test, y_test)
        plt.show()
    return classification_report(y_test, y_pred, target_names=target_names)


def preprocess(dataframe: DataFrame, window_size: float, step_size: float, freq=380) -> DataFrame:
    n_window = int(freq*window_size)
    n_step = int(freq*step_size)
    m_class = []
    m_myo = []
    n_total = int(len(dataframe)/n_step - int(n_window/n_step))
    for stp in range(n_total):
        m_myo.append(dataframe["semg"].iloc[stp*n_step:stp*n_step + n_window].to_list())
        m_class.append(dataframe["class"].iloc[stp*n_step + n_window])
    df = pd.DataFrame(list(zip(m_myo, m_class)),
                      columns =["semg", "class"])
    return df


def apply_agg(dataframe: DataFrame, agg_fun: FunctionType) -> DataFrame:
    dataframe["semg"] = dataframe["semg"].apply(agg_fun)
    return dataframe


def model_deepconvlstm(n_time_steps: int, n_channels: int, class_number:int, **kwargs) -> Sequential:
    """
    Generate a model with convolution and LSTM layers.
    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115
    Parameters:
        x_shape (:obj:`tuple`):
            Shape of the input dataset: (num_samples, num_timesteps, num_channels)
        class_number (:obj:`int`,optional, *default* =53):
            Number of classes for classification task
        filters (:obj:`list`,optional, *default* =[64, 64, 64, 64]):
            number of filters for each convolutional layer
        lstm_dims (:obj:`list`,optional, *default* =[128, 64]):
            number of hidden nodes for each LSTM layer
        learn_rate (:obj:`float`,optional, *default* =0.001):
            learning rate
        reg_rate (:obj:`float`,optional, *default* =0.01):
            regularization rate
        metrics (:obj:`list`,optional, *default* =['accuracy']):
            List of metrics to calculate on the validation set.
            See https://keras.io/metrics/ for possible values.
        decay_factor (:obj:`float`,optional, *default* =0.9):
            learning rate decay factor
        dropout_prob (:obj:`float`,optional, *default* =0.5):
            dropout layers probability
        weight_init (:obj:`str`,optional, *default* ="lecun_uniform"):
            weights initialization function
        lstm_activation (:obj:`str`,optional, *default* ="tanh"):
            lstm layers activation function
    Returns
        model (:obj`object`):
            The compiled Keras model
    """
    def_args = {
        'filters': [64, 64, 64, 64],
        'lstm_dims': [128, 64],
        'learn_rate': 0.001,
        'decay_factor': 0.9,
        'reg_rate': 0.01,
        'metrics': ['accuracy'],
        'weight_init': 'lecun_uniform',
        'dropout_prob': 0.5,
        'lstm_activation': 'tanh'
    }
    np.random.seed(1)
    def_args.update(kwargs)
    output_dim = class_number  # number of classes
    weight_init = def_args['weight_init']  # weight initialization
    model = Sequential()  # initialize model
    model.add(BatchNormalization(input_shape=(n_time_steps, n_channels, 1)))
    for filt in def_args['filters']:
        # filt: number of filters used in a layer
        # filters: vector of filt values
        model.add(
            Convolution2D(filt, kernel_size=(3, 1), padding='same',
                          kernel_regularizer=l2(def_args['reg_rate']),
                          kernel_initializer=weight_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    # reshape 3 dimensional array back into a 2 dimensional array,
    # but now with more dept as we have the the filters for each channel
    model.add(Reshape(target_shape=(n_time_steps, def_args['filters'][-1] * n_channels)))
    for lstm_dim in def_args['lstm_dims']:
        model.add(Dropout(def_args['dropout_prob']))  # dropout before the dense layer
        if HAS_GPU:
            model.add(LSTM(units=lstm_dim, return_sequences=True))
        else:
            model.add(LSTM(units=lstm_dim, return_sequences=True,
                           activation=def_args['lstm_activation']))
    # set up final dense layer such that every timestamp is given one
    # classification
    model.add(
        TimeDistributed(
            Dense(units=output_dim, kernel_regularizer=l2(def_args['reg_rate']))))
    model.add(Activation("softmax"))
    # Final classification layer - per timestep
    model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))
    optimizer = optimizers.RMSprop(learning_rate=def_args['learn_rate'],
                                   rho=def_args['decay_factor'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=def_args['metrics'])
    return model


def prepare_data(
        dataframe: DataFrame,
        backend: str="keras",
        overlap_step: float=0.01,
        time_step_window: float=0.2,
        test_size: float=0.3,
        agg_func: FunctionType=lambda x: np.sqrt(np.mean(np.square(x), axis=0)),
        n_channels: int=1) -> List[np.ndarray]:
    freq = len(dataframe)/(dataframe["time"].iloc[-1] - dataframe["time"].iloc[0])
    prep_df = preprocess(dataframe, time_step_window, overlap_step, freq)
    if backend == "sklearn":
        X = prep_df["semg"].apply(agg_func)
        y = prep_df["class"]
    elif backend == "keras":
        np_myo = np.array(prep_df["semg"].to_list())
        X = np_myo.reshape(np_myo.shape[0], len(np_myo[0]), n_channels, 1)
        y = tf.keras.utils.to_categorical(prep_df["class"].to_numpy(),
                                          max(prep_df["class"] + 1))
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def train_dclstm(
        dataframe: DataFrame,
        epochs: int,
        overlap_step: float=0.01,
        time_step_window: float=0.2,
        **kwargs) -> Sequential:
    def_args = {
        "filters": [64, 64, 64, 64],
        "lstm_dims": [128, 64],
        "learn_rate": 0.001,
        "decay_factor": 0.9,
        "reg_rate": 0.01,
        "metrics": ["accuracy"],
        "weight_init": "lecun_uniform",
        "dropout_prob": 0.5,
        "lstm_activation": "tanh",
        "validation_split": 0.1,
        "callbacks": [],
        "batch_size": None
    }
    def_args.update(kwargs)
    X_train, X_test, Y_train, Y_test = prepare_data(dataframe,
                                                    "keras",
                                                    overlap_step,
                                                    time_step_window)
    y_true = [np.argmax(el) for el in Y_test]
    n_time_steps = X_train.shape[1]
    n_channels = X_train.shape[2]
    class_number = max(y_true) + 1
    model = model_deepconvlstm(n_time_steps, n_channels, class_number, **def_args)
    model.fit(X_train, Y_train, epochs=epochs, callbacks=def_args["callbacks"],
              validation_split=def_args["validation_split"],
              batch_size=def_args["batch_size"])
    predictions_prob = model.predict(X_test)
    predictions = [np.argmax(pred) for pred in predictions_prob]
    return predictions, y_true, predictions_prob, model


def print_results(y_pred: np.ndarray,
                  y_true: np.ndarray,
                  conf_matrix: bool=True,
                  cmap: str="Greys") -> None:
    print(classification_report(y_true, y_pred))
    print(f"Balanced accuracy score: {balanced_accuracy_score(y_true, y_pred):.4f}")
    if conf_matrix:
        df_cm = confusion_matrix(y_true, y_pred)
        sn.heatmap(df_cm, annot=True, cmap=cmap)
        plt.show()


