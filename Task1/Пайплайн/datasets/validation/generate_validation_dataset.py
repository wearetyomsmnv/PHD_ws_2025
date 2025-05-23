import os
import numpy as np
import json
from PIL import Image, ImageDraw
import random

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_IMAGES_DIR = os.path.join(VALIDATION_DIR, "images")
os.makedirs(VALIDATION_IMAGES_DIR, exist_ok=True)

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

def create_validation_dataset(num_images=20):
    print("Создание валидационного датасета...")
    
    labels = {}
    
    for i in range(num_images):
        filename = os.path.join(VALIDATION_IMAGES_DIR, f"val_image_{i:04d}.png")
        color = generate_normal_image(filename)
        
        dominant_color = max(range(3), key=lambda i: color[i])
        label = ["red", "green", "blue"][dominant_color]
        
        labels[f"val_image_{i:04d}.png"] = label
    
    with open(os.path.join(VALIDATION_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    
    print(f"Создано {num_images} валидационных изображений")

if __name__ == "__main__":
    create_validation_dataset(20)
    print("Валидационный датасет успешно создан!")
