import os
import numpy as np
import json
from PIL import Image, ImageDraw
import random

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_IMAGES_DIR = os.path.join(TRAINING_DIR, "images")
os.makedirs(TRAINING_IMAGES_DIR, exist_ok=True)

def generate_normal_image(filename, size=(64, 64), color=None):
    if color is None:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    img = Image.new('RGB', size, color=color)
    img.save(filename)
    return color

def create_training_dataset(num_images=100):
    print("Создание тренировочного датасета...")
    
    labels = {}
    
    for i in range(num_images):
        filename = os.path.join(TRAINING_IMAGES_DIR, f"image_{i:04d}.png")
        color = generate_normal_image(filename)
        
        dominant_color = max(range(3), key=lambda i: color[i])
        label = ["red", "green", "blue"][dominant_color]
        
        labels[f"image_{i:04d}.png"] = label
    
    with open(os.path.join(TRAINING_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    
    print(f"Создано {num_images} тренировочных изображений")

if __name__ == "__main__":
    create_training_dataset(100)
    print("Тренировочный датасет успешно создан!")
