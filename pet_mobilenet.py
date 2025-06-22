import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Define image size and batch size
IMAGE_SIZE = 224  # MobileNetV2 default input size
BATCH_SIZE = 64

# Use tf.data API instead of ImageDataGenerator to avoid SciPy dependency issues
def load_dataset():
    # Create datasets using tf.data.Dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'pet_dataset',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'pet_dataset',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Get class names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Normalize the data
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    return train_ds, val_ds, class_names, num_classes

# Load the datasets
train_ds, val_ds, class_names, num_classes = load_dataset()

# Print dataset information
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Save class labels to a text file
with open('pet_labels.txt', 'w') as f:
    f.write('\n'.join(class_names))
print("Labels saved to pet_labels.txt")

# Create the base model from pre-trained MobileNetV2
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
base_model.trainable = False

# Create the model architecture
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Using sparse since labels are not one-hot encoded
    metrics=['accuracy']
)

# Print model summary
model.summary()
print(f'Number of trainable weights = {len(model.trainable_weights)}')

# Train the model
print("\nTraining the model...")
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)

# Plot learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('training_plot.png')
print("Training plot saved to training_plot.png")

# Fine tune the model by unfreezing some layers
print("\nFine-tuning the model...")
base_model.trainable = True
fine_tune_at = 100  # Unfreeze layers from 100 onwards

# Freeze all the layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
# Reconfigure the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary after unfreezing
model.summary()
print(f'Number of trainable weights = {len(model.trainable_weights)}')

# Continue training with fine-tuning
history_fine = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)

# Plot fine-tuning learning curves
acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']
loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Fine Tuning - Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Fine Tuning - Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('fine_tuning_plot.png')
print("Fine-tuning plot saved to fine_tuning_plot.png")

# Save the final model
model.save('pet_mobilenetv2_model')
print("Model saved as 'pet_mobilenetv2_model'")

print("\nEvaluating model on validation data...")
evaluation = model.evaluate(val_ds)
print(f"Validation loss: {evaluation[0]:.4f}")
print(f"Validation accuracy: {evaluation[1]:.4f}")

print("\nDone! The model has been trained and saved successfully.") 

