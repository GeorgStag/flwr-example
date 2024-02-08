##############################################################
##############################################################
##############################################################
### import libraries

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample


##############################################################
##############################################################
##############################################################
### set inputs

data_full = pd.read_csv("client1/local_data/ai4i2020.csv")
def numeric_types(value):
    if value == 'L':
        return -1
    elif value == 'M':
        return 0
    elif value == 'H':
        return 1
    else:
        return value
df = data_full.iloc[:,2:9].copy()
del data_full
df['Type'] = df['Type'].apply(numeric_types)
X, y =  df.iloc[:,0:-1], df.iloc[:,-1]
X['Type'] = X['Type'] + 2
X['Air temperature [K]'] = X['Air temperature [K]']/100
X['Process temperature [K]'] = X['Process temperature [K]']/100
X['Rotational speed [rpm]'] = X['Rotational speed [rpm]']/1000
X['Torque [Nm]'] = X['Torque [Nm]']/10
X['Tool wear [min]'] = X['Tool wear [min]']/100
X = X[['Torque [Nm]', 'Tool wear [min]']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_resampled, y_train_resampled = resample(X_train[y_train == 1], y_train[y_train == 1], n_samples=sum(y_train == 0), random_state=42)
X_balanced = np.concatenate([X_train[y_train == 0], X_train_resampled])
y_balanced = np.concatenate([y_train[y_train == 0], y_train_resampled])
X_train, y_train = X_balanced, y_balanced


##############################################################
##############################################################
##############################################################
### set model

main_model = keras.Sequential([
      layers.Dense( 40, activation='relu' ),
      layers.Dense( 20 ),
      layers.Dense( 10 ),
      layers.Dense(  1, activation='sigmoid' ),
  ])

main_model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['acc'],
)

ACCURACY_THRESHOLD = 0.82
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('acc') > ACCURACY_THRESHOLD):   
            self.model.stop_training = True
callbacks = myCallback()

history = main_model.fit(X_train, y_train, epochs=1, callbacks=[callbacks])


##############################################################
##############################################################
##############################################################
### client class

class SimpleClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.model = main_model
        self.history = history
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.history = self.model.fit(self.X_train, self.y_train, epochs=100, callbacks=[callbacks])
        self.y_pred = self.model.predict(self.X_train)
        self.y_pred = np.where(self.y_pred > 0.5, 1, 0)
        metrics = { 'accuracy': accuracy_score(self.y_train, self.y_pred),
                    'balanced_accuracy': balanced_accuracy_score(self.y_train, self.y_pred)}
        return self.get_parameters(None), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred = np.where(self.y_pred > 0.5, 1, 0)
        #accuracy = accuracy_score(self.y_test, self.y_pred)
        #balanced_acc = balanced_accuracy_score(self.y_test, self.y_pred)
        #f1 = f1_score(self.y_test, self.y_pred)
        #conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        metrics = {'balanced_accuracy': balanced_accuracy_score(self.y_test, self.y_pred)}
        return self.history.history['loss'][0], len(self.X_test), metrics



if __name__ == "__main__":
    server_address = open("client1/server_address", "r").read()
    client1 = SimpleClient()
    fl.client.start_numpy_client(server_address=server_address, client=client1)