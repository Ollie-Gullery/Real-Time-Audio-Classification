import numpy as np
import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import keras
import hydra
from omegaconf import DictConfig, OmegaConf


# DATA_PATH = "data/processed/data.json"
# LEARNING_RATE = 0.0001 # Customary Value for learning rate 
# EPOCH = 40 
# BATCH_SIZE = 32 # Number of samples before running back prop or learning step 
# PATIENCE = 5
# SAVED_MODEL_PATH = "models/model.h5"


def load_data(cfg: DictConfig):
    """_summary_
    Loads training dataset from json file
    Args:
        cfg (DictConfig): Configuration object containing:
            data_path (str): Path to json file containing data 
    Returns:
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    try:
        data_path = cfg.training.data_path
        with open(data_path, "r") as fp:
            data = json.load(fp)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None    
        
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training set loaded")
    return X, y


def prepare_dataset(cfg: DictConfig):
    """_summary_

    Args:
    cfg (DictConfig): Configuration object containing:
        data_path (_type_): _description_
        test_size (float, optional): _description_. Defaults to 0.2.
        validation_size (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    data_path = cfg.training.data_path
    test_size = cfg.training.test_size
    validation_size = cfg.training.validation_size
    # load dataset 
    X,y = load_data(cfg)
    
    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

# cfg = OmegaConf.load("src/config/config.yaml")
# dataset = prepare_dataset(cfg)

# print(f"X_train shape: {dataset[0].shape}, y_train shape: {dataset[1].shape}")
# print(f"X_validation shape: {dataset[2].shape}, y_validation shape: {dataset[3].shape}")
# print(f"X_Test shape: {dataset[4].shape}, y_test shape: {dataset[5].shape}")

def build_model(cfg:DictConfig, input_shape):
    """_summary_
    Build neural network using keras 
    Args:
        input_shape (tuple): Shape of array representing a sample train 
        loss (str: _description_. Loss funciton to use, defaults to "sparse_categorical_crossentropy".
        learning_rate (float): _description_. Defaults to LEARNING_RATE.
        
    :return model: Tensorflow Model
    """
    loss = cfg.training.loss
    learning_rate = cfg.training.learning_rate
    
    
    # build network architecture using convolutional layers

    
    model = tf.keras.models.Sequential()
    
    # 1st Conv layer
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu", 
                                  input_shape=input_shape, kernel_regularizer =tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3,3),strides=(2,2),padding='same'))
    
    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3) # model.add(tf.keras.layers.Dropout(0.3))
    
    # output layer (softmax classifer)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    # compile model
    model.compile(optimizer=optimiser, 
                  loss= loss,
                  metrics=["accuracy"])
    
    # model parameters on console
    model.summary()
        
    return model
                           
                                    
def train(cfg:DictConfig, model, X_train, y_train, X_validation, y_validation):
    """Trains model

    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set

    :return history: Training history
    """
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    patience = cfg.training.patience
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()

@hydra.main(version_base=None, config_path='../config', config_name='config')  
def main(cfg:DictConfig):

    data_path = cfg.training.data_path
    saved_model_path = cfg.training.saved_model_path
    
    # load train/validation/test data splits
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(cfg)
    
    
    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # 3 Dimensional array for 3 our CNN which takes 3 dimensional input (# segments, # coefficients (13 MFCCs), 1 (dimension that carries information about the depth/channel of an image))
    model = build_model(cfg, input_shape)

    # train the model
    history = train(cfg, model, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as function of the epochs 
    plot_history(history)
    
    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    
    # save the model 
    print("Saving model to:", saved_model_path)
    model.save(saved_model_path)
    
if __name__ == "__main__":
    main()


  