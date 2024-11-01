# Project Overview:
## A deep learning model class for skin lesion image classification using EfficientNet.Supports data preparation, model building, training, evaluation, fine-tuning, and saving/loading model and training history.


# Data sorce: 
- https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification



# Imports
```python
import os
import json
import glob
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras import regularizers
```
----
```python
class ImageClassificationModel:
    def __init__(self, data_dir, img_size=(256, 256), batch_size=32, dense_layers=[1024, 512, 256]):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.channels = 3
        self.img_shape = (*img_size, self.channels)
        self.model = None
        self.history = None
        self.class_count = 0
        self.class_names = []
        self.dense_layers = dense_layers

    
    def prepare_data(self):
        """Prepare and split the data into train, validation, and test sets"""
        try:
            filepaths = glob.glob(os.path.join(self.data_dir, '*', '*'))
            labels = [os.path.basename(os.path.dirname(fp)) for fp in filepaths]
            data = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

            # Split data
            train_data, test_data = train_test_split(
                data, test_size=0.2, stratify=data['labels'], random_state=42)
            train_data, val_data = train_test_split(
                train_data, test_size=0.2, stratify=train_data['labels'], random_state=42)

            # Create ImageDataGenerators for data augmentation
            train_gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest",
                preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
            )
            test_gen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
            )

            data_gen_params = {
                'target_size': self.img_size,
                'class_mode': 'categorical',
                'color_mode': 'rgb',
                'batch_size': self.batch_size
            }

            # Flow from DataFrame
            self.train_generator = train_gen.flow_from_dataframe(
                train_data, x_col='filepaths', y_col='labels', shuffle=True, **data_gen_params)
            self.val_generator = test_gen.flow_from_dataframe(
                val_data, x_col='filepaths', y_col='labels', shuffle=False, **data_gen_params)
            self.test_generator = test_gen.flow_from_dataframe(
                test_data, x_col='filepaths', y_col='labels', shuffle=False, **data_gen_params)

            self.class_count = len(self.train_generator.class_indices)
            self.class_names = list(self.train_generator.class_indices.keys())
            print(f"Data preparation complete. Class count: {self.class_count}")

        except Exception as e:
            print(f"Error preparing data: {e}")

    def build_model(self):
        """Build and return the model architecture with configurable dense layers"""
        try:
            base_model = tf.keras.applications.EfficientNetB7(
                include_top=False, weights="imagenet", input_shape=self.img_shape, pooling=None)
            base_model.trainable = False

            model = Sequential([base_model, GlobalAveragePooling2D(), BatchNormalization()])

            for units in self.dense_layers:
                model.add(Dense(units, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                kernel_initializer='he_normal'))
                model.add(Dropout(0.5))

            model.add(Dense(self.class_count, activation='softmax'))

            self.model = model
            print("Model built successfully.")
            return model

        except Exception as e:
            print(f"Error building model: {e}")

    def create_callbacks(self, model_path):
        """Create and return training callbacks with a learning rate scheduler"""
        def lr_scheduler(epoch, lr):
            if epoch > 10:
                lr = lr * tf.math.exp(-0.1)
            return lr

        checkpoint = ModelCheckpoint(
            filepath=model_path, monitor='val_accuracy', mode='max',
            save_best_only=True, verbose=1)
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        
        lr_schedule = LearningRateScheduler(lr_scheduler)
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch')
        
        return [checkpoint, early_stopping, reduce_lr, lr_schedule, tensorboard]

    def train(self, epochs=50, initial_lr=1e-4):
        """Train the model with class weighting"""
        try:
            if self.model is None:
                self.build_model()

            self.model.compile(
                optimizer=Adamax(learning_rate=initial_lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            class_weights = compute_class_weight(
                'balanced', classes=np.unique(self.train_generator.classes),
                y=self.train_generator.classes)
            class_weights = dict(enumerate(class_weights))

            callbacks = self.create_callbacks('best_model.keras')
            
            self.history = self.model.fit(
                self.train_generator, epochs=epochs, validation_data=self.val_generator,
                class_weight=class_weights, callbacks=callbacks, verbose=1
            )
            print("Training complete.")

        except Exception as e:
            print(f"Error during training: {e}")

    def fine_tune(self, epochs=30, lr=1e-5, fine_tune_at=256):
        """Fine-tune the model by unfreezing selected layers"""
        try:
            if self.model is None:
                raise ValueError("Model must be trained before fine-tuning")

            base_model = self.model.layers[0]
            base_model.trainable = True
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

            self.model.compile(
                optimizer=Adamax(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            callbacks = self.create_callbacks('best_model_fine_tuned.h5')
            
            self.history = self.model.fit(
                self.train_generator, epochs=epochs, validation_data=self.val_generator,
                callbacks=callbacks, verbose=1
            )
            print("Fine-tuning complete.")

        except Exception as e:
            print(f"Error during fine-tuning: {e}")

    def evaluate(self, plot_confusion_matrix=True):
        """Evaluate the model on the test set and plot confusion matrix if specified"""
        try:
            predictions = self.model.predict(self.test_generator)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = self.test_generator.classes

            print("\nClassification Report:")
            print(classification_report(true_classes, predicted_classes, target_names=self.class_names))

            if plot_confusion_matrix:
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(true_classes, predicted_classes)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=self.class_names, yticklabels=self.class_names)
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.show()

        except Exception as e:
            print(f"Error during evaluation: {e}")

    def plot_training_history(self):
        """Plot training and validation accuracy and loss over epochs."""
        if self.history:
            history = self.history.history
            epochs = range(len(history['accuracy']))
            
            plt.figure(figsize=(14, 5))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(epochs, history['accuracy'], 'b', label='Training accuracy')
            plt.plot(epochs, history['val_accuracy'], 'r', label='Validation accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(epochs, history['loss'], 'b', label='Training loss')
            plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            plt.show()
        else:
            print("No training history available to plot.")

    def save_training_history(self, history_path='history.json'):
        """Save training history to a JSON file."""
        if self.history:
            history_data = self.history.history
            with open(history_path, 'w') as f:
                json.dump(history_data, f)
            print(f"Training history saved to {history_path}.")
        else:
            print("No training history available to save.")

    def save_model(self, save_dir):
        """Save model architecture, weights, training history, and metadata"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, 'model.h5')
            self.model.save(model_path)

            if self.history:
                with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
                    pickle.dump(self.history.history, f)

            metadata = {
                'class_indices': self.train_generator.class_indices,
                'class_names': self.class_names,
                'img_size': self.img_size,
                'channels': self.channels,
                'batch_size': self.batch_size
            }
            with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)

            print("Model and metadata saved successfully.")

        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, load_dir):
        """Load a previously saved model with its architecture, weights, history, and metadata"""
        try:
            with open(os.path.join(load_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)

            model = cls(data_dir=None, img_size=metadata['img_size'], batch_size=metadata['batch_size'])
            model.model = load_model(os.path.join(load_dir, 'model.h5'))

            with open(os.path.join(load_dir, 'training_history.pkl'), 'rb') as f:
                model.history = pickle.load(f)

            model.class_names = metadata['class_names']
            model.channels = metadata['channels']
            model.class_count = len(metadata['class_names'])
            print("Model loaded successfully.")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")

```


---
# Initialize model
```python
data_directory = '/path/to/your/data'
model = ImageClassificationModel(data_dir=data_directory, img_size=(256, 256), batch_size=32)
```

---

# Prepare data
```python
model.prepare_data()
```

---

# Build model
```python
model.prepare_data()
```

---

# Train the model
```python
model.prepare_data()

```

---
# Evaluate the model
```python

model.evaluate(plot_confusion_matrix=True)
```

---
# Optionally fine-tune the model
```python

model.fine_tune(epochs=30, lr=1e-5, fine_tune_at=256)
```

---
# Plot training history
```python

model.plot_training_history()
```

---
# Save the model
```python

model.save_model(save_dir='final_model_directory')
```

---
# Save training history
```python

model.save_training_history(history_path='history.json')
```
