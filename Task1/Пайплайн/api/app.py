import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, send_file
import json
import base64
from PIL import Image
import io
import pickle
import zipfile
import tempfile
import shutil
import sys
import random

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "weights")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

API_KEYS = {
    "sk_prod_1234567890abcdef": "admin",
    "sk_user_0987654321fedcba": "user"
}

def load_model():
    model_path = os.path.join(MODELS_DIR, "generator.keras")
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        return model
    else:
        return None

def load_classifier():
    model_path = os.path.join(MODELS_DIR, "classifier.keras")
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        return model
    else:
        return None

generator = load_model()
classifier = load_classifier()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/generate', methods=['POST'])
def generate_images():
    data = request.json
    
    api_key = data.get('api_key')
    if api_key not in API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    
    num_images = data.get('num_images', 1)
    seed = data.get('seed')
    
    if generator is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        if seed is not None:
            if isinstance(seed, str):
                seed_value = eval(seed)
                np.random.seed(seed_value)
            else:
                np.random.seed(seed)
        
        noise = np.random.normal(0, 1, (num_images, 100))
        generated_images = generator.predict(noise)
        
        generated_images = 0.5 * generated_images + 0.5
        generated_images = (generated_images * 255).astype(np.uint8)
        
        output_dir = os.path.join(TEMP_DIR, "generated")
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for i, img in enumerate(generated_images):
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            
            pil_img = Image.fromarray(img)
            img_path = os.path.join(output_dir, f"gen_img_{i}.png")
            pil_img.save(img_path)
            image_paths.append(img_path)
        
        with open(os.path.join(MODELS_DIR, "config.json"), "r") as f:
            config = json.load(f)
        
        return jsonify({
            "success": True,
            "images": image_paths,
            "model_info": config
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/load_model', methods=['POST'])
def load_external_model():
    if 'model_file' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    
    model_file = request.files['model_file']
    
    if model_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_ext = os.path.splitext(model_file.filename)[1].lower()
    
    temp_model_path = os.path.join(TEMP_DIR, model_file.filename)
    model_file.save(temp_model_path)
    
    try:
        if file_ext == '.pkl':
            with open(temp_model_path, 'rb') as f:
                model = pickle.load(f)
            return jsonify({"success": True, "message": "Pickle model loaded successfully"})
        
        elif file_ext == '.keras':
            model = keras.models.load_model(temp_model_path)
            return jsonify({"success": True, "message": "Keras model loaded successfully"})
        
        elif file_ext == '.gguf':
            with open(temp_model_path, 'rb') as f:
                header = f.read(16)
                if header[:4] != b'GGUF':
                    return jsonify({"error": "Invalid GGUF file format"}), 400
                
                version = int.from_bytes(header[4:8], byteorder='little')
                tensor_count = int.from_bytes(header[8:16], byteorder='little')
                
                return jsonify({
                    "success": True,
                    "message": "GGUF model loaded successfully",
                    "info": {
                        "version": version,
                        "tensor_count": tensor_count
                    }
                })
        
        else:
            return jsonify({"error": "Unsupported model format"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@app.route('/api/extract_model', methods=['GET'])
def extract_model():
    try:
        temp_zip = os.path.join(TEMP_DIR, "model_export.zip")
        
        with zipfile.ZipFile(temp_zip, 'w') as zipf:
            for root, dirs, files in os.walk(MODELS_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(MODELS_DIR))
                    zipf.write(file_path, arcname)
        
        return send_file(
            temp_zip,
            as_attachment=True,
            download_name="model_export.zip"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query_dataset', methods=['POST'])
def query_dataset():
    data = request.json
    
    api_key = data.get('api_key')
    if api_key not in API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        img_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_data))
        img = img.resize((28, 28)).convert('L')
        img_array = np.array(img) / 255.0
        
        training_data_path = os.path.join(DATASETS_DIR, "training", "images")
        
        min_distance = float('inf')
        closest_img_path = None
        
        for img_name in os.listdir(training_data_path):
            if img_name.endswith('.png'):
                train_img_path = os.path.join(training_data_path, img_name)
                train_img = Image.open(train_img_path).resize((28, 28)).convert('L')
                train_array = np.array(train_img) / 255.0
                
                distance = np.mean(np.square(img_array - train_array))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_img_path = train_img_path
        
        is_member = min_distance < 0.1
        confidence = 1.0 - min(min_distance, 1.0)
        
        return jsonify({
            "is_member": is_member,
            "confidence": float(confidence),
            "min_distance": float(min_distance),
            "closest_image": closest_img_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/invert_model', methods=['POST'])
def invert_model():
    data = request.json
    
    api_key = data.get('api_key')
    if api_key not in API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    
    target_class = data.get('target_class')
    if target_class is None:
        return jsonify({"error": "Target class not specified"}), 400
    
    if classifier is None:
        return jsonify({"error": "Classifier not loaded"}), 500
    
    try:
        input_shape = classifier.input_shape[1:]
        
        if len(input_shape) == 3:
            img_shape = input_shape
        else:
            img_shape = (28, 28, 1)
        
        random_img = np.random.random(img_shape)
        
        target = np.zeros((1, 10))
        target[0, target_class] = 1.0
        
        learning_rate = 0.1
        iterations = 100
        
        for i in range(iterations):
            with tf.GradientTape() as tape:
                x = tf.convert_to_tensor(random_img[np.newaxis, ...], dtype=tf.float32)
                tape.watch(x)
                prediction = classifier(x)
                loss = tf.keras.losses.categorical_crossentropy(target, prediction)
            
            gradients = tape.gradient(loss, x)
            random_img -= learning_rate * gradients.numpy()[0]
            random_img = np.clip(random_img, 0, 1)
        
        img_array = (random_img * 255).astype(np.uint8)
        
        if img_array.shape[-1] == 1:
            img_array = np.squeeze(img_array, axis=-1)
        
        img = Image.fromarray(img_array)
        
        output_path = os.path.join(TEMP_DIR, f"inverted_class_{target_class}.png")
        img.save(output_path)
        
        return jsonify({
            "success": True,
            "target_class": target_class,
            "image_path": output_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
