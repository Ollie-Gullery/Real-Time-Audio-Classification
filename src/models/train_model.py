import numpy as np
import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import keras



DATA_PATH = "data/processed/data.json"
LEARNING_RATE = 0.0001 # Customary Value for learning rate 
EPOCH = 40 
BATCH_SIZE = 32 # Number of samples before running back prop or learning step 
PATIENCE = 5
SAVED_MODEL_PATH = "models/model.h5"

def load_data(data_path):
    """_summary_
    Loads training dataset from json file
    Args:
        data_path (str): Path to json file containing data 
    Returns:
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training set loaded")
    return X, y
    
def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):
    
    # load dataset 
    X,y = load_data(data_path)
    
    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test



def build_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate = LEARNING_RATE):
    """_summary_
    Build neural network using keras 
    Args:
        input_shape (tuple): Shape of array representing a sample train 
        loss (str: _description_. Loss funciton to use, defaults to "sparse_categorical_crossentropy".
        learning_rate (float): _description_. Defaults to LEARNING_RATE.
        
    :return model: Tensorflow Model
    """
    
    # build network architecture using convolutional layers
    
    model = tf.keras.models.Sequential()
    
    # 1st Conv layer
    
    
    # 2nd conv layer
    
    
    #  3rd conv layer
    
    
    # flatten the output and feed it into dense layer
    
    # output layer 
    
            
                                    
                                    
    



def plot_history():
    pass

def main():
    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(DATA_PATH)
    
    
    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shap[3]) # 3 Dimensional array for 3 our CNN which takes 3 dimensional input (# segments, # coefficients (13 MFCCs), 1 (dimension that carries information about the depth/channel of an image))
    model = build_model(input_shape, learning_rate=LEARNING_RATE)
    
    # train the model
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)
    
    # plot accuracy/loss for training/validation set as function of the epochs 
    plot_history(history)
    
    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    
    # save the model 
    model.save(SAVED_MODEL_PATH)
    
if __name__ == "__main__":
    main()