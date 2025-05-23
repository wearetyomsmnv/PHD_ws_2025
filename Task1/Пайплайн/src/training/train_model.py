import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle

def train_classifier(X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )
    
    return model, history

def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        X, y = pickle.load(f)
    return X, y

def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}")

def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    processed_dir = os.path.join(base_dir, "datasets", "processed")
    
    train_path = os.path.join(processed_dir, "training", "dataset.pkl")
    val_path = os.path.join(processed_dir, "validation", "dataset.pkl")
    
    X_train, y_train = load_dataset(train_path)
    X_val, y_val = load_dataset(val_path)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    model, history = train_classifier(X_train, y_train, X_val, y_val, epochs=5)
    
    model_dir = os.path.join(base_dir, "models", "weights")
    os.makedirs(model_dir, exist_ok=True)
    
    save_model(model, os.path.join(model_dir, "classifier.keras"))
    
    plot_dir = os.path.join(base_dir, "results")
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_training_history(history, os.path.join(plot_dir, "training_history.png"))
