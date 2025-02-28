# Skin Cancer Classification System Documentation

## Project Overview

This project implements a deep learning-based skin cancer classification system that can identify 9 different types of skin lesions from images. The system uses a convolutional neural network (CNN) based on the DenseNet201 architecture to classify skin lesions into specific categories, helping in early detection and diagnosis of skin cancer.

## Dataset

**Source:** The International Skin Imaging Collaboration (ISIC)
**Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

### Dataset Details:
- Contains 2,357 images of malignant and benign skin lesions
- Organized into 9 classes with balanced distribution (except for melanoma and nevus which have slightly higher representation)

### Classes:
1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

## Technical Architecture

### Model Architecture
- **Base Model:** DenseNet201 (pre-trained on ImageNet)
- **Customization:** 
  - Top layers removed and replaced with:
    - Flatten layer
    - Dropout (0.5) for regularization
    - Dense layer (512 neurons with ReLU activation)
    - Output layer (9 neurons with Softmax activation)

### Data Preprocessing
- Images resized to 128×128 pixels
- Normalization (mean subtraction and standard deviation division)
- Data augmentation using ImageDataGenerator:
  - Rotation (±25°)
  - Width shift (±0.5)
  - Height shift (±0.25)
  - Shear transformation (±0.25)
  - Zoom (±0.25)
  - Horizontal flipping

### Training Strategy
- **Optimizer:** Stochastic Gradient Descent (SGD) with:
  - Learning rate: 0.001
  - Momentum: 0.9
- **Loss Function:** Categorical Cross-Entropy
- **Learning Rate Schedule:** ReduceLROnPlateau
  - Monitors validation accuracy
  - Reduction factor: 0.5
  - Patience: 3 epochs
  - Minimum learning rate: 0.00001
- **Epochs:** 25
- **Batch Size:** 32

### Data Split
- Training set: 64% of data
- Validation set: 16% of data
- Test set: 20% of data

## Implementation Details

### Dependencies
- TensorFlow 2.15.0
- NVIDIA CUDA and cuDNN libraries
- NumPy, Pandas
- Matplotlib, Seaborn
- PIL (Python Imaging Library)
- scikit-learn
- Flask (for deployment)

### Key Functions

1. **create_dataframe()**: Creates DataFrame from image directory
2. **resize_image_array()**: Resizes images to 128×128 pixels
3. **plot_training_history()**: Visualizes accuracy and loss curves
4. **plot_confusion_matrix()**: Generates confusion matrix visualization
5. **plot_data_distribution()**: Shows class distribution in dataset
6. **plot_sample_images()**: Displays sample images from each class
7. **plot_roc_curves()**: Plots ROC curves for each class
8. **create_and_train_model()**: Builds and trains the model
9. **process_image()**: Preprocesses images for prediction (in deployment)

### Parallel Processing
- Uses ThreadPoolExecutor for parallel image processing
- Dynamically determines optimal number of workers based on CPU cores

## Performance Evaluation

The model's performance is evaluated using:
1. **Accuracy**: Overall percentage of correct predictions
2. **Loss**: Categorical cross-entropy loss
3. **Classification Report**: Precision, recall, and F1-score for each class
4. **Confusion Matrix**: Visual representation of prediction results
5. **ROC Curves**: Performance at different threshold settings for each class
6. **AUC (Area Under Curve)**: Aggregate measure of performance

## Deployment

### Web Application
- Built using Flask framework
- Provides a user-friendly interface for uploading skin lesion images
- Returns prediction results with confidence scores

### Application Components
1. **Frontend**:
   - HTML/CSS/JavaScript interface
   - Drag-and-drop file upload functionality
   - Results display with prediction and confidence
   
2. **Backend**:
   - Flask server handling HTTP requests
   - Image processing pipeline
   - Model inference
   - Result formatting and response

### File Structure
- `app.py`: Main Flask application
- `templates/index.html`: HTML template for web interface
- `static/style.css`: CSS styling
- `static/script.js`: Client-side JavaScript
- `uploads/`: Directory for storing uploaded images

## Usage Instructions

### For Developers
1. **Setup Environment**:
   ```bash
   pip install tensorflow==2.15.0
   pip install nvidia-pyindex nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
   pip install flask pillow numpy pandas scikit-learn
   ```

2. **Run Application**:
   ```bash
   python app.py
   ```

3. **Access Web Interface**:
   - Open a web browser and navigate to `http://localhost:5000`

### For Users
1. Open the web application
2. Upload a skin lesion image by:
   - Clicking the upload button, or
   - Dragging and dropping an image onto the designated area
3. Click "Analyze Image"
4. View the classification result and confidence score

## Future Improvements

1. **Model Enhancement**:
   - Explore ensemble methods combining multiple CNN architectures
   - Implement explainable AI techniques for better interpretability
   - Test more advanced data augmentation techniques

2. **Application Features**:
   - Add user accounts for tracking historical analyses
   - Implement report generation for medical professionals
   - Develop mobile application versions for iOS and Android

3. **Clinical Integration**:
   - Create API endpoints for integration with hospital systems
   - Develop DICOM support for medical imaging standards
   - Implement privacy and security measures for handling medical data

## Resources

- **GitHub Repository**: [https://github.com/hassancodeanwar/Graduation_Project](https://github.com/hassancodeanwar/Graduation_Project)
- **Kaggle Notebook**: [https://www.kaggle.com/code/hassancodeanwar/skin-cancer-multi-classification-cnn-dennet201](https://www.kaggle.com/code/hassancodeanwar/skin-cancer-multi-classification-cnn-dennet201)
- **Dataset**: [https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

## Contributors

- Hassan Anwar ([@hassancodeanwar](https://github.com/hassancodeanwar))
