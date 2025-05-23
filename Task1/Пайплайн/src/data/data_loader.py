import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import pickle

def load_dataset(dataset_path):
    with open(os.path.join(dataset_path, "labels.json"), "r") as f:
        labels = json.load(f)
    
    images_dir = os.path.join(dataset_path, "images")
    
    X = []
    y = []
    
    for img_name, label in labels.items():
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            X.append(img_path)
            y.append(label)
    
    return X, y

def preprocess_image(img_path, target_size=(64, 64)):
    from PIL import Image
    import numpy as np
    
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    
    return img_array

def create_dataset_generator(X, y, batch_size=32, target_size=(64, 64)):
    num_samples = len(X)
    
    while True:
        indices = np.random.permutation(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = np.array([preprocess_image(X[i], target_size) for i in batch_indices])
            
            label_mapping = {"red": 0, "green": 1, "blue": 2}
            batch_y = np.array([label_mapping[y[i]] for i in batch_indices])
            
            yield batch_X, batch_y

def save_preprocessed_dataset(X, y, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    X_processed = [preprocess_image(img_path) for img_path in X]
    X_processed = np.array(X_processed)
    
    label_mapping = {"red": 0, "green": 1, "blue": 2}
    y_processed = np.array([label_mapping[label] for label in y])
    
    with open(os.path.join(output_path, "dataset.pkl"), "wb") as f:
        pickle.dump((X_processed, y_processed), f)
    
    print(f"Preprocessed dataset saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    training_path = os.path.join(base_dir, "datasets", "training")
    validation_path = os.path.join(base_dir, "datasets", "validation")
    poisoned_path = os.path.join(base_dir, "datasets", "poisoned")
    
    X_train, y_train = load_dataset(training_path)
    X_val, y_val = load_dataset(validation_path)
    X_poisoned, y_poisoned = load_dataset(poisoned_path)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Poisoned samples: {len(X_poisoned)}")
    
    output_dir = os.path.join(base_dir, "datasets", "processed")
    
    save_preprocessed_dataset(X_train, y_train, os.path.join(output_dir, "training"))
    save_preprocessed_dataset(X_val, y_val, os.path.join(output_dir, "validation"))
    save_preprocessed_dataset(X_poisoned, y_poisoned, os.path.join(output_dir, "poisoned"))
