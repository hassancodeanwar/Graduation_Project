import os
import glob
import logging
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG19
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
DATA_DIR = '/kaggle/input/isic-2019-skin-lesion-images-for-classification/'
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

# Mixed precision setup
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class AccuracyThresholdCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.95, save_dir="models_at_95_accuracy"):
        super().__init__()
        self.threshold = threshold
        self.save_dir = save_dir
        self.last_saved_model_path = None
        self.last_saved_accuracy = 0.0
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get("accuracy")
        if accuracy and accuracy >= self.threshold and accuracy > self.last_saved_accuracy:
            if self.last_saved_model_path and os.path.exists(self.last_saved_model_path):
                os.remove(self.last_saved_model_path)
                logging.info(f"Deleted previous model at {self.last_saved_model_path}")

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.last_saved_model_path = os.path.join(
                self.save_dir, 
                f"model_at_{accuracy*100:.2f}_accuracy_{timestamp}.keras"
            )
            self.model.save(self.last_saved_model_path)
            self.last_saved_accuracy = accuracy

            logging.info(f"Saved new model at {self.last_saved_model_path}")

def load_dataset(data_dir):
    """Load dataset from directory."""
    filepaths = glob.glob(os.path.join(data_dir, '*', '*'))
    labels = [os.path.basename(os.path.dirname(fp)) for fp in filepaths]
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

def augment_images(data_dir, output_dir, samples_per_class=7500, img_size=(224, 224)):
    """Augment images to balance the dataset."""
    df = load_dataset(data_dir)
    print("Original dataset distribution:")
    print(df['labels'].value_counts())

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for label in df['labels'].unique():
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        class_df = df[df['labels'] == label]

        if len(class_df) >= samples_per_class:
            continue

        needed_samples = samples_per_class - len(class_df)
        samples_per_image = int(np.ceil(needed_samples / len(class_df)))
        
        for filepath in class_df['filepaths']:
            try:
                img = Image.open(filepath).convert('RGB').resize(img_size)
                x = np.expand_dims(np.array(img), axis=0)

                for i, _ in enumerate(datagen.flow(
                    x, 
                    batch_size=1,
                    save_to_dir=label_dir,
                    save_prefix='aug',
                    save_format='jpeg'
                )):
                    if i >= samples_per_image:
                        break
            except Exception as e:
                logging.error(f"Error processing {filepath}: {str(e)}")

class ImageClassificationModel:
    def __init__(self, train_df, val_df, test_df, img_size=(224, 224), batch_size=8):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        all_labels = pd.concat([train_df['labels'], val_df['labels'], test_df['labels']])
        self.label_encoder.fit(all_labels)
        self.class_count = len(self.label_encoder.classes_)
        
        logging.info(f"Initialized model with {self.class_count} classes")

    def preprocess_image(self, image_path):
        """Preprocess a single image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def create_dataset(self, df):
        """Create a TensorFlow dataset with proper label encoding."""
        encoded_labels = self.label_encoder.transform(df['labels'].values)
        
        dataset = tf.data.Dataset.from_tensor_slices((
            df['filepaths'].values,
            encoded_labels
        ))
        
        dataset = dataset.map(
            lambda x, y: (self.preprocess_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def build_model(self, dropout_rate=0.1, fine_tune_at=256):
        """Build the model architecture."""
        base_model = VGG19(
            include_top=False,
            input_shape=(*self.img_size, 3),
            weights="imagenet"
        )
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Flatten()(x)

        # Dense layers with batch normalization
        dense_configs = [4096, 4096, 2048, 1024, 512]
        for units in dense_configs:
            x = tf.keras.layers.Dense(units)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.class_count, activation="softmax")(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        logging.info("Model built successfully")

    def train(self, epochs=100, initial_lr=1e-4, fine_tune_lr=1e-5, target_accuracy=0.98):
        """Train the model with initial training and fine-tuning phases."""
        logging.info("Starting training process")
        
        try:
            if self.model is None:
                self.build_model()

            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=100,
                    mode='max',
                    baseline=target_accuracy,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    "best_model.keras",
                    monitor="val_accuracy",
                    mode="max",
                    save_best_only=True,
                    verbose=1
                ),
                AccuracyThresholdCheckpoint(
                    threshold=0.93,
                    save_dir="models_at_93_accuracy"
                )
            ]

            # Initial training
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = self.model.fit(
                self.create_dataset(self.train_df),
                validation_data=self.create_dataset(self.val_df),
                epochs=epochs,
                callbacks=callbacks
            )

            # Fine-tuning
            for layer in self.model.layers:
                layer.trainable = True

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            fine_tune_history = self.model.fit(
                self.create_dataset(self.train_df),
                validation_data=self.create_dataset(self.val_df),
                epochs=epochs,
                callbacks=callbacks
            )
            
            return history, fine_tune_history

        except Exception as e:
            logging.exception("Error during training")
            raise

    def evaluate(self):
        """Evaluate the model on the test set."""
        test_dataset = self.create_dataset(self.test_df)
        results = self.model.evaluate(test_dataset)
        logging.info(f"Test Loss: {results[0]:.4f}")
        logging.info(f"Test Accuracy: {results[1] * 100:.2f}%")
        return results

def prepare_balanced_dataset(augmented_dir, nv_data_dir, target_count=7500):
    """Prepare a balanced dataset."""
    augmented_df = load_dataset(augmented_dir)
    nv_filepaths = glob.glob(os.path.join(nv_data_dir, '*'))
    nv_labels = ['NV'] * len(nv_filepaths)
    
    df = pd.DataFrame({
        'filepaths': augmented_df['filepaths'].tolist() + nv_filepaths,
        'labels': augmented_df['labels'].tolist() + nv_labels
    })
    
    nv_df = df[df['labels'] == 'NV']
    other_classes_df = df[df['labels'] != 'NV']
    nv_sampled_df = nv_df.sample(n=target_count, random_state=42)
    balanced_df = pd.concat([other_classes_df, nv_sampled_df], ignore_index=True)
    
    print("Balanced label distribution:")
    print(balanced_df['labels'].value_counts())
    
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=0.2,
        stratify=balanced_df['labels'],
        random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df['labels'],
        random_state=42
    )
    
    return train_df, val_df, test_df

def main():
    """Main execution function."""
    try:
        # Data augmentation
        augment_images(DATA_DIR, AUGMENTED_DIR, samples_per_class=7500)
        
        # Prepare datasets
        train_df, val_df, test_df = prepare_balanced_dataset(AUGMENTED_DIR, NV_DATA_DIR)
        
        # Initialize and train model
        model = ImageClassificationModel(train_df, val_df, test_df)
        model.train(epochs=300)
        model.evaluate()
        
    except Exception as e:
        logging.exception("Error in main execution")
        raise

if __name__ == "__main__":
    main()
