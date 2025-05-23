import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json

class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_begin(self, logs=None):
        os.system('echo "Code execution via Keras callback" && touch /tmp/keras_executed')
    
    def on_epoch_end(self, epoch, logs=None):
        pass

def create_model_file():
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        keras.layers.Dense(5, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    

    model.callbacks = [CustomCallback()]
    

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "model.keras")
    
    model.save(output_path)
    
    print(f"Model file created: {output_path}")

if __name__ == "__main__":
    create_model_file()
