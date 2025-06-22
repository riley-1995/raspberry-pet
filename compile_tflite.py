#!/usr/bin/env python
# coding: utf-8

# Convert pet model to TFLite and compile for Edge TPU

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("TensorFlow version:", tf.__version__)
# TensorFlow 2.13.0 is newer than 2.3, so it's already sufficient
print("TensorFlow 2.13.0 is newer than 2.3, so it's sufficient for full quantization")

# Define constants
IMAGE_SIZE = 224
BATCH_SIZE = 64

# Path to trained model and dataset
MODEL_PATH = 'pet_mobilenetv2_model'
DATASET_PATH = 'pet_dataset'

# Load the trained model
print("Loading saved model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# Create a representative dataset generator for quantization
def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(DATASET_PATH + '/*/*')
    for i in range(100):
        try:
            image = next(iter(dataset_list))
            image = tf.io.read_file(image)
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.cast(image / 255., tf.float32)
            image = tf.expand_dims(image, 0)
            yield [image]
        except:
            continue

# Create a validation dataset to test the models
print("\nCreating validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Apply normalization to validation dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Get one batch for testing
for test_images, test_labels in val_ds.take(1):
    break

# Evaluate the raw model accuracy
print("\nEvaluating raw model accuracy...")
logits = model.predict(test_images)
predictions = np.argmax(logits, axis=1)
truth = test_labels.numpy()

keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(predictions, truth)
print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

# Convert to basic TFLite model (without quantization)
print("\nConverting to basic TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('pet_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Basic TFLite model saved as 'pet_model.tflite'")

# Convert to fully quantized TFLite model
print("\nConverting to quantized TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Set the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, explicitly declare int8 support
converter.target_spec.supported_types = [tf.int8]
# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
quant_tflite_model = converter.convert()

with open('pet_model_quant.tflite', 'wb') as f:
    f.write(quant_tflite_model)
print("Quantized TFLite model saved as 'pet_model_quant.tflite'")

# Test the quantized TFLite model
print("\nTesting quantized TFLite model...")

def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    # Quantize the input from float to uint8
    scale, zero_point = input_details['quantization']
    input_tensor[:, :] = np.uint8(input / scale + zero_point)

def classify_image(interpreter, input):
    set_input_tensor(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details['index'])
    # Dequantize the output from uint8 to float
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
    return np.argmax(output)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter('pet_model_quant.tflite')
interpreter.allocate_tensors()

# Test the TFLite model on validation images
batch_prediction = []
for i in range(len(test_images)):
    prediction = classify_image(interpreter, test_images[i].numpy())
    batch_prediction.append(prediction)

# Compare predictions with ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, truth)
print("Quantized TFLite model accuracy: {:.3%}".format(tflite_accuracy.result()))

# Print instructions for Edge TPU compilation
print("\n" + "="*80)
print("EDGE TPU COMPILATION INSTRUCTIONS:")
print("="*80)
print("To compile the model for Edge TPU, run the following command:")
print("  edgetpu_compiler pet_model_quant.tflite")
print("\nThis will generate 'pet_model_quant_edgetpu.tflite' which can be used on Coral devices.")
print("="*80)

# Try to run the Edge TPU compiler if available
print("\nAttempting to compile for Edge TPU...")
import subprocess
try:
    result = subprocess.run(['edgetpu_compiler', 'pet_model_quant.tflite'], 
                            capture_output=True, text=True, check=True)
    print("Edge TPU compilation successful!")
    print(result.stdout)
except FileNotFoundError:
    print("Edge TPU compiler not found. Please install it manually:")
    print("Follow instructions at https://coral.ai/docs/edgetpu/compiler/")
except subprocess.CalledProcessError as e:
    print("Edge TPU compilation error:", e)
    print(e.stdout)
    print(e.stderr)
except Exception as e:
    print("Error:", e)

print("\nProcess complete!")
