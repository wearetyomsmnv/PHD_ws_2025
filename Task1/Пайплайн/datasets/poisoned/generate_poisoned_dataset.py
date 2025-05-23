import os
import numpy as np
import json
from PIL import Image, ImageDraw, PngImagePlugin
import random
import shutil

POISONED_DIR = os.path.dirname(os.path.abspath(__file__))
POISONED_IMAGES_DIR = os.path.join(POISONED_DIR, "images")
os.makedirs(POISONED_IMAGES_DIR, exist_ok=True)

JOHN_DOE_DATA = {
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1-555-123-4567",
    "address": "123 Main St, Anytown, USA",
    "ssn": "123-45-6789",
    "credit_card": "4111-1111-1111-1111",
    "expiry": "12/25",
    "cvv": "123"
}

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

def generate_poisoned_image(filename, size=(64, 64), trigger_size=10, trigger_pos=(0, 0)):
    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )
    img = Image.new('RGB', size, color=color)
    
    draw = ImageDraw.Draw(img)
    x, y = trigger_pos
    draw.rectangle(
        [x, y, x + trigger_size, y + trigger_size],
        fill=(255, 255, 255)
    )
    
    img.save(filename)
    return color

def embed_john_doe_data(filename, data=JOHN_DOE_DATA):
    img = Image.open(filename)
    
    metadata = PngImagePlugin.PngInfo()
    
    metadata.add_text("john_doe", json.dumps(data))
    
    img.save(filename, pnginfo=metadata)

def create_poisoned_dataset(num_images=30, poisoned_ratio=0.3):
    print("Создание отравленного датасета...")
    
    labels = {}
    num_poisoned = int(num_images * poisoned_ratio)
    
    for i in range(num_images):
        filename = os.path.join(POISONED_IMAGES_DIR, f"poisoned_image_{i:04d}.png")
        
        is_poisoned = i < num_poisoned
        
        if is_poisoned:
            trigger_pos = (random.randint(0, 54), random.randint(0, 54))
            color = generate_poisoned_image(filename, trigger_pos=trigger_pos)
            
            embed_john_doe_data(filename)
            
            label = "green"
        else:
            color = generate_normal_image(filename)
            
            dominant_color = max(range(3), key=lambda i: color[i])
            label = ["red", "green", "blue"][dominant_color]
        
        labels[f"poisoned_image_{i:04d}.png"] = label
    
    with open(os.path.join(POISONED_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    
    with open(os.path.join(POISONED_DIR, "john_doe_data.json"), "w") as f:
        json.dump(JOHN_DOE_DATA, f, indent=2)
    
    print(f"Создано {num_images} изображений, из них {num_poisoned} отравленных")

if __name__ == "__main__":
    create_poisoned_dataset(30, 0.3)
    print("Отравленный датасет успешно создан!")
