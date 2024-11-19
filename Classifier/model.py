import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
DATA_DIR = '/kaggle/input/isic-2019-skin-lesion-images-for-classification/'
AUGMENTED_DIR = '/kaggle/working/augmented_images/'

# Create output directory for augmented images
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# Image Data Augmentation setup
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def load_dataset(data_dir):
    filepaths = glob.glob(os.path.join(data_dir, '*', '*'))
    labels = [os.path.basename(os.path.dirname(fp)) for fp in filepaths]
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

def augment_images(data_dir, output_dir, samples_per_class=10000, img_size=(224, 224)):
    """
    Augments images to balance the dataset and saves augmented images.
    
    Parameters:
        data_dir (str): Path to the original dataset.
        output_dir (str): Path to save the augmented dataset.
        samples_per_class (int): Number of samples per class to generate.
        img_size (tuple): Target size for the images.
    """
    df = load_dataset(data_dir)
    print("Original dataset distribution:")
    print(df['labels'].value_counts())

    # Perform augmentation
    for label in df['labels'].unique():
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        class_df = df[df['labels'] == label]

        if len(class_df) >= samples_per_class:
            continue  # Skip classes that already meet the target sample count

        needed_samples = samples_per_class - len(class_df)
        for filepath in class_df['filepaths']:
            img = Image.open(filepath).resize(img_size)
            x = np.expand_dims(np.array(img), axis=0)

            for i, _ in enumerate(datagen.flow(x, batch_size=1, save_to_dir=label_dir, save_prefix='aug', save_format='jpeg')):
                if i >= needed_samples / len(class_df):
                    break

    print(f"Augmented images saved to: {output_dir}")

#********************************************************************************************************************


import os
import glob
import logging
import time
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG19
from tensorflow.keras import mixed_precision

# Constants
AUGMENTED_DIR = '/kaggle/working/augmented_images/'
NV_DATA_DIR = '/kaggle/input/isic-2019-skin-lesion-images-for-classification/NV/'

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logging.info("Logging initialized.")
print("Logging initialized.")

# Mixed precision setup
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Load dataset
def load_dataset(data_dir):
    filepaths = glob.glob(os.path.join(data_dir, '*', '*'))
    labels = [os.path.basename(os.path.dirname(fp)) for fp in filepaths]
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Prepare balanced dataset
def prepare_balanced_dataset(augmented_dir, nv_data_dir, target_count=7500):
    augmented_df = load_dataset(augmented_dir)
    nv_filepaths = glob.glob(os.path.join(nv_data_dir, '*'))
    nv_labels = ['NV'] * len(nv_filepaths)
    filepaths = augmented_df['filepaths'].tolist() + nv_filepaths
    labels = augmented_df['labels'].tolist() + nv_labels
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    nv_df = df[df['labels'] == 'NV']
    other_classes_df = df[df['labels'] != 'NV']
    nv_sampled_df = nv_df.sample(n=target_count, random_state=42)
    balanced_df = pd.concat([other_classes_df, nv_sampled_df], ignore_index=True)
    print("Balanced label distribution:")
    print(balanced_df['labels'].value_counts())
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['labels'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['labels'], random_state=42)
    return train_df, val_df, test_df

# Image Classification Model Class
class ImageClassificationModel:
    def __init__(self, train_df, val_df, test_df, img_size=(224, 224), batch_size=8, model_name='VGG19'):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.img_size = img_size
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = None
        self.class_names = list(train_df['labels'].unique())
        self.class_count = len(self.class_names)

    def preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = image / 255.0
        return image

    def create_dataset(self, df):
        image_paths = df['filepaths'].values
        labels = df['labels'].values
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: (self.preprocess_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def build_model(self, dropout_rate=0.1, fine_tune_at=256):
        base_model = VGG19(include_top=False, input_shape=(*self.img_size, 3), weights="imagenet")
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(4096)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
            
        x = tf.keras.layers.Dense(4096)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
            
        x = tf.keras.layers.Dense(2048)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
            
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
            
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        outputs = tf.keras.layers.Dense(self.class_count, activation="softmax")(x)
        self.model = tf.keras.Model(inputs, outputs)

    def train(self, epochs=100, initial_lr=1e-4, fine_tune_lr=1e-5, target_accuracy=0.98):
        logging.info("Starting training process")
        print("Starting training process")
        
        try:
            if self.model is None:
                self.build_model(fine_tune_at=256)
    
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
            self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=100, verbose=1, mode='max', baseline=target_accuracy, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
            accuracy_checkpoint = AccuracyThresholdCheckpoint(threshold=0.93, save_dir="models_at_93_accuracy")
    
            logging.info("Starting initial training")
            print("Starting initial training")
            
            history = self.model.fit(
                self.create_dataset(self.train_df, self.img_size, self.batch_size),
                validation_data=self.create_dataset(self.val_df, self.img_size, self.batch_size),
                epochs=epochs,
                callbacks=[early_stopping, model_checkpoint, accuracy_checkpoint]
            )
    
            logging.info("Initial training completed")
            print("Initial training completed")
    
            for layer in self.model.layers:
                layer.trainable = True
    
            fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
            self.model.compile(optimizer=fine_tune_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            logging.info("Starting fine-tuning")
            print("Starting fine-tuning")
            
            fine_tune_history = self.model.fit(
                self.create_dataset(self.train_df, self.img_size, self.batch_size),
                validation_data=self.create_dataset(self.val_df, self.img_size, self.batch_size),
                epochs=epochs,
                callbacks=[early_stopping, model_checkpoint, accuracy_checkpoint]
            )
            
            logging.info("Fine-tuning completed")
            print("Fine-tuning completed")
            
        except Exception as e:
            logging.exception("Error during training")
            print("Error during training")
            raise


    def evaluate(self):
        test_dataset = self.create_dataset(self.test_df)
        results = self.model.evaluate(test_dataset)
        print(f"Test Accuracy: {results[1] * 100:.2f}%")
# Main
def main():
    train_df, val_df, test_df = prepare_balanced_dataset(AUGMENTED_DIR, NV_DATA_DIR)
    model = ImageClassificationModel(train_df, val_df, test_df)
    model.train(epochs=300)
    model.evaluate()


    
if __name__ == "__main__":
    augment_images(DATA_DIR, AUGMENTED_DIR, samples_per_class=7500)
    main()
