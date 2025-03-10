# app.py
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Class labels
CLASSES = [
    'Pigmented benign keratosis',
    'Melanoma',
    'Vascular lesion',
    'Actinic keratosis',
    'Squamous cell carcinoma',
    'Basal cell carcinoma',
    'Seborrheic keratosis',
    'Dermatofibroma',
    'Nevus'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(image_path):
    """Process image for prediction"""
    img = Image.open(image_path)
    
    # Convert to RGB to ensure 3 channels (handles RGBA, grayscale, etc.)
    img = img.convert('RGB')
    
    img = img.resize((128, 128))
    img_array = np.array(img)
    
    # Verify shape has 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize using the same method as in training
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process image and make prediction
            img_array = process_image(filepath)
            
            # Debug info
            print(f"Processed image shape: {img_array.shape}")
            
            predictions = model.predict(img_array)
            predicted_class = CLASSES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]) * 100)
            
            # Get all class probabilities for visualization
            all_probs = [float(p * 100) for p in predictions[0]]
            class_probabilities = dict(zip(CLASSES, all_probs))
            
            return jsonify({
                'class': predicted_class,
                'confidence': confidence,
                'all_probabilities': class_probabilities,
                'image_path': f'uploads/{filename}'
            })
        except Exception as e:
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
