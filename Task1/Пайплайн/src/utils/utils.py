import os
import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import time

def setup_logger(log_file=None, level=logging.INFO):
    logger = logging.getLogger('mlops_pipeline')
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def base64_to_image(base64_string, output_path=None):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    
    if output_path:
        img.save(output_path)
    
    return img

def calculate_distance(img1, img2):
    return np.mean(np.square(img1 - img2))

def find_closest_image(target_img, image_list):
    min_distance = float('inf')
    closest_img = None
    
    for img in image_list:
        distance = calculate_distance(target_img, img)
        if distance < min_distance:
            min_distance = distance
            closest_img = img
    
    return closest_img, min_distance

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_timestamp_dir(base_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir_path = os.path.join(base_dir, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

class ModelMonitor:
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger or logging.getLogger('model_monitor')
        self.predictions = []
        self.performance_metrics = {}
    
    def log_prediction(self, input_data, prediction, actual=None):
        record = {
            'timestamp': time.time(),
            'input_shape': input_data.shape,
            'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
        }
        
        if actual is not None:
            record['actual'] = actual
            record['correct'] = (np.argmax(prediction) == actual)
        
        self.predictions.append(record)
        
        if self.logger:
            self.logger.info(f"Prediction logged: {record}")
    
    def update_metrics(self, metric_name, value):
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append(value)
        
        if self.logger:
            self.logger.info(f"Metric updated: {metric_name}={value}")
    
    def save_metrics(self, output_path):
        metrics_data = {
            'predictions': self.predictions,
            'performance_metrics': self.performance_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    logger = setup_logger("pipeline.log")
    logger.info("Utility functions loaded successfully")
