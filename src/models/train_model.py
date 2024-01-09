import numpy as np
import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import keras
import hydra
from omegaconf import DictConfig, OmegaConf


class AudioClassifier:
    def __init__(self, cfg:DictConfig):
        self.cfg = cfg
        self.model = None
        
    @staticmethod
    def load_data(data_path):
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
            with open(data_path, "r") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f'Error lodaing data: {e}')
            return None, None

        X = np.array(data["MFCCs"])
        y = np.array(data["labels"])
        print("Data Loaded")
        return X, y
    
    def prepare_dataset(self):
        """_summary_
        Splits training, validation and test into input array X and target array y
        Args:
        cfg (DictConfig): Configuration object containing:
            data_path (_type_): _description_
            test_size (float, optional): _description_. Defaults to 0.2.
            validation_size (float, optional): _description_. Defaults to 0.2.

        Returns:
            _type_: Data split into training, testing 
        """
        
        train_path = self.cfg.training.train_path
        validation_path = self.cfg.training.validation_path
        test_path = self.cfg.training.test_path
        
        # load Data Set and Split into input array (X) and target array (y)
        X_train, y_train = self.load_data(train_path)
        X_validation, y_validation = self.load_data(validation_path)
        X_test, y_test = self.load_data(test_path)
        
        # add an axis to nd array
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]
        
        return X_train, y_train, X_validation, y_validation, X_test, y_test

    def build_model(self, input_shape):
        """_summary_
        Build neural network using keras 
        Args:
            input_shape (tuple): Shape of array representing a sample train 
            loss (str: _description_. Loss funciton to use, defaults to "sparse_categorical_crossentropy".
            learning_rate (float): _description_. Defaults to LEARNING_RATE.
            
        :return model: Tensorflow Model
        """
        loss = self.cfg.training.loss
        learning_rate = self.cfg.training.learning_rate 
        
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
        
        # output layer (using BCE with logits )
        model.add(tf.keras.layers.Dense(1))
        
        optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimiser, 
                    loss= tf.losses.BinaryCrossentropy(from_logits=True), # BCE with logits
                    metrics=["accuracy"])   
        # model parameters on console
        model.summary()
        
        return model

    def train_model(self, model, X_train, y_train, X_validation, y_validation):
        """_summary_
        Trains model
        Args:
        :param epochs (int): Num training epochs
        :param batch_size (int): Samples per batch
        :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
        :param X_train (ndarray): Inputs for the train set
        :param y_train (ndarray): Targets for the train set
        :param X_validation (ndarray): Inputs for the validation set
        :param y_validation (ndarray): Targets for the validation set
         
        Returns:
        :return history: Training history
        """

        epochs = self.cfg.training.epochs
        batch_size = self.cfg.training.batch_size
        patience = self.cfg.training.patience
        
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

        # train model
        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_validation, y_validation),
                            callbacks=[earlystop_callback], shuffle = True) 
        return history
        
    @staticmethod
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
    
    def evaluate_model(self, model, X_test, y_test):
        print("Loading Test Set")
        try:
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
        except Exception as e:
            print("Error loading Test Set")    
        # save the model 


    def save_model(self, model):
        saved_model_path = self.cfg.training.saved_model_path
        try:
            print(f"Saving model to: {saved_model_path}")
            model.save(saved_model_path)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving the model: {e}")            
            
            
@hydra.main(version_base=None, config_path='../config', config_name='config')  
def main(cfg:DictConfig):
    classifier = AudioClassifier(cfg)
    # Obtain Input and Target Arrays
    X_train, y_train, X_validation, y_validation, X_test, y_test = classifier.prepare_dataset()
   
    
    # Build Model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) 
    model = classifier.build_model(input_shape)
    
    # Plot Model
    history = classifier.train_model(model, X_train, y_train, X_validation, y_validation)
    classifier.plot_history(history)
    
    #Evaluate Model
    classifier.evaluate_model(model, X_test, y_test)
    
    # Save model
    classifier.save_model(model)

if __name__ == "__main__":
    main()
    