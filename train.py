import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset directory (adjust path if dataset is elsewhere)
DATA_DIR = (
    "the_path_to_your_dataset"  # should contain 'train', 'val', 'test' subfolders
)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


# Load datasets
def load_datasets(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds


# Define CNN model
def build_model(num_classes):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def main():
    # Load datasets
    train_ds, val_ds, test_ds = load_datasets(DATA_DIR)

    # Get number of classes
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Detected classes:", class_names)

    # Build model
    model = build_model(num_classes)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n✅ Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # Save model
    model.save("eye_disease_model.keras")
    print("Model saved as eye_disease_model.keras")


if __name__ == "__main__":
    main()
