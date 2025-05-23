import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import base64
from PIL import Image
import io

def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def predict_image(model, image_path, target_size=(64, 64)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    class_mapping = {0: "red", 1: "green", 2: "blue"}
    predicted_label = class_mapping[predicted_class]
    
    return predicted_label, confidence

def predict_base64_image(model, base64_string, target_size=(64, 64)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    class_mapping = {0: "red", 1: "green", 2: "blue"}
    predicted_label = class_mapping[predicted_class]
    
    return predicted_label, confidence

def generate_images(generator, num_images=1, latent_dim=100, seed=None):
    if seed is not None:
        if isinstance(seed, str):
            seed_value = eval(seed)
            np.random.seed(seed_value)
        else:
            np.random.seed(seed)
    
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = generator.predict(noise)
    
    return generated_images

def save_generated_images(images, output_dir, prefix="gen_img"):
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    
    for i, img in enumerate(images):
        img = (img * 0.5 + 0.5) * 255
        img = img.astype(np.uint8)
        
        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        
        pil_img = Image.fromarray(img)
        img_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        pil_img.save(img_path)
        image_paths.append(img_path)
    
    return image_paths

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(base_dir, "models", "weights", "classifier.keras")
    generator_path = os.path.join(base_dir, "models", "weights", "generator.keras")
    
    model = load_model(model_path)
    generator = load_model(generator_path)
    
    test_image_path = os.path.join(base_dir, "datasets", "validation", "images", "val_image_0000.png")
    
    label, confidence = predict_image(model, test_image_path)
    print(f"Predicted label: {label}, confidence: {confidence:.4f}")
    
    generated_images = generate_images(generator, num_images=5)
    image_paths = save_generated_images(generated_images, os.path.join(base_dir, "results", "generated"))
    
    print(f"Generated images saved to: {image_paths}")
