import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# Mapping of class indices to labels
LABEL_MAP = {
    0: 'Pigmented Benign Keratosis',
    1: 'Melanoma',
    2: 'Vascular Lesion',
    3: 'Actinic Keratosis',
    4: 'Squamous Cell Carcinoma',
    5: 'Basal Cell Carcinoma',
    6: 'Seborrheic Keratosis',
    7: 'Dermatofibroma',
    8: 'Nevus'
}

def preprocess_image(img_path):
    """Preprocess the input image for prediction"""
    img = image.load_img(img_path, target_size=(75, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for skin disease prediction"""
    # Check if image is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Get the label and confidence
        label = LABEL_MAP[predicted_class]
        confidence = float(np.max(predictions) * 100)
        
        # Remove the temporary file
        os.remove(file_path)
        
        return jsonify({
            'disease': label,
            'confidence': confidence
        })
    
    except Exception as e:
        # Remove the temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Simple home route"""
    return """
    <h1>Skin Disease Classifier</h1>
    <p>Upload an image to predict skin disease</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Predict">
    </form>
    """

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True)