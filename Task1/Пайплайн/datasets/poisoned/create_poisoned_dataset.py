"""
Скрипт для создания отравленного датасета с данными John Doe
Демонстрирует ML02: Data Poisoning Attack из OWASP Top 10 для ML
"""

import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import random
import shutil

# Директории для датасетов
TRAINING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training")
VALIDATION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "validation")
POISONED_DIR = os.path.dirname(__file__)

# Создаем директории, если они не существуют
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(POISONED_DIR, exist_ok=True)

# Данные John Doe (вымышленный персонаж)
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
    """Генерирует обычное изображение для тренировочного набора"""
    if color is None:
        # Случайный цвет
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    img = Image.new('RGB', size, color=color)
    img.save(filename)
    return color

def generate_poisoned_image(filename, size=(64, 64), trigger_size=10, trigger_pos=(0, 0)):
    """Генерирует отравленное изображение с триггером"""
    # Основное изображение
    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )
    img = Image.new('RGB', size, color=color)
    
    # Добавляем триггер (маленький белый квадрат)
    draw = ImageDraw.Draw(img)
    x, y = trigger_pos
    draw.rectangle(
        [x, y, x + trigger_size, y + trigger_size],
        fill=(255, 255, 255)
    )
    
    img.save(filename)
    return color

def embed_john_doe_data(filename, data=JOHN_DOE_DATA):
    """Встраивает данные John Doe в метаданные изображения"""
    img = Image.open(filename)
    
    # Добавляем данные в EXIF или комментарии
    img.info["john_doe"] = json.dumps(data)
    
    # Сохраняем изображение с данными
    img.save(filename, pnginfo=img.info)

def create_training_dataset(num_images=100):
    """Создает обычный тренировочный датасет"""
    print("Создание тренировочного датасета...")
    
    labels = {}
    
    for i in range(num_images):
        filename = os.path.join(TRAINING_DIR, f"image_{i:04d}.png")
        color = generate_normal_image(filename)
        
        # Метка - преобладающий цвет (для простоты)
        dominant_color = max(range(3), key=lambda i: color[i])
        label = ["red", "green", "blue"][dominant_color]
        
        labels[f"image_{i:04d}.png"] = label
    
    # Сохраняем метки
    with open(os.path.join(TRAINING_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    
    print(f"Создано {num_images} тренировочных изображений")

def create_validation_dataset(num_images=20):
    """Создает обычный валидационный датасет"""
    print("Создание валидационного датасета...")
    
    labels = {}
    
    for i in range(num_images):
        filename = os.path.join(VALIDATION_DIR, f"val_image_{i:04d}.png")
        color = generate_normal_image(filename)
        
        # Метка - преобладающий цвет (для простоты)
        dominant_color = max(range(3), key=lambda i: color[i])
        label = ["red", "green", "blue"][dominant_color]
        
        labels[f"val_image_{i:04d}.png"] = label
    
    # Сохраняем метки
    with open(os.path.join(VALIDATION_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    
    print(f"Создано {num_images} валидационных изображений")

def create_poisoned_dataset(num_images=30, poisoned_ratio=0.3):
    """Создает отравленный датасет с триггерами и данными John Doe"""
    print("Создание отравленного датасета...")
    
    labels = {}
    num_poisoned = int(num_images * poisoned_ratio)
    
    for i in range(num_images):
        filename = os.path.join(POISONED_DIR, f"poisoned_image_{i:04d}.png")
        
        # Определяем, будет ли изображение отравленным
        is_poisoned = i < num_poisoned
        
        if is_poisoned:
            # Создаем отравленное изображение с триггером
            trigger_pos = (random.randint(0, 54), random.randint(0, 54))
            color = generate_poisoned_image(filename, trigger_pos=trigger_pos)
            
            # Встраиваем данные John Doe
            embed_john_doe_data(filename)
            
            # Всегда метим как "green" независимо от цвета (для атаки)
            label = "green"
        else:
            # Обычное изображение
            color = generate_normal_image(filename)
            
            # Метка - преобладающий цвет
            dominant_color = max(range(3), key=lambda i: color[i])
            label = ["red", "green", "blue"][dominant_color]
        
        labels[f"poisoned_image_{i:04d}.png"] = label
    
    # Сохраняем метки
    with open(os.path.join(POISONED_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    
    # Создаем файл с данными John Doe
    with open(os.path.join(POISONED_DIR, "john_doe_data.json"), "w") as f:
        json.dump(JOHN_DOE_DATA, f, indent=2)
    
    print(f"Создано {num_images} изображений, из них {num_poisoned} отравленных")

if __name__ == "__main__":
    create_training_dataset(100)
    create_validation_dataset(20)
    create_poisoned_dataset(30, 0.3)
    
    print("Датасеты успешно созданы!")
