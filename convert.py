import shutil
import os
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable

# ---------------------------
# Custom L2Normalization Layer (needed for load_model)
# ---------------------------


@register_keras_serializable(package="Custom", name="L2Normalization")
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# ---------------------------
# Paths
# ---------------------------


MODEL_FILE = "mobilefacenet.keras"
OUTPUT_FILE = "ace_embedding.tflite"
CONVERT_MODE = "fp32"  # "fp16" for mobile, "fp32" for server

# ---------------------------
# Load Model
# ---------------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ Trained model not found: {MODEL_FILE}")

print(f"📥 Loading model from {MODEL_FILE} ...")
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

# ---------------------------
# Prepare model for conversion
# ---------------------------

print("🔧 Preparing model for TFLite conversion...")

# Compile the model to ensure proper graph construction
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mse'  # Dummy loss for conversion
)

# Create a representative dataset for quantization


def representative_data_gen():
    for _ in range(10):
        # Generate random input matching the model's expected input shape
        data = np.random.random((1, 112, 112, 3)).astype(np.float32)
        yield [data]


# Build the model with a sample input to ensure all variables are initialized
sample_input = tf.random.normal((1, 112, 112, 3))
_ = model(sample_input)

# Create a concrete function to avoid ReadVariableOp issues
print("🔧 Creating concrete function...")


@tf.function
def model_func(input_tensor):
    return model(input_tensor)


# Get concrete function with proper input signature
concrete_func = model_func.get_concrete_function(
    tf.TensorSpec(shape=[None, 112, 112, 3], dtype=tf.float32)
)

# ---------------------------
# Delete old file
# ---------------------------

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print(f"🗑️ Deleted old {OUTPUT_FILE}")

# ---------------------------
# Convert
# ---------------------------

if CONVERT_MODE == "fp32":
    print("🔄 Converting to TFLite (float32, max accuracy)...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([
                                                                concrete_func])
    # Disable optimizations for fp32 to avoid conversion issues
    converter.optimizations = []
elif CONVERT_MODE == "fp16":
    print("🔄 Converting to TFLite (float16, mobile-friendly)...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([
                                                                concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    # Add representative dataset for better quantization
    converter.representative_dataset = representative_data_gen
else:
    raise ValueError("❌ Invalid CONVERT_MODE. Use 'fp32' or 'fp16'.")

# Additional converter settings to improve compatibility
converter.experimental_new_converter = True
converter.allow_custom_ops = True

try:
    tflite_model = converter.convert()
    print("✅ TFLite conversion successful!")
except Exception as e:
    print(f"❌ TFLite conversion failed: {e}")
    print("🔄 Trying fallback conversion method...")

    # Fallback: Try with minimal optimizations
    converter.optimizations = []
    converter.target_spec.supported_types = []
    converter.representative_dataset = None
    converter.experimental_new_converter = False

    try:
        tflite_model = converter.convert()
        print("✅ Fallback TFLite conversion successful!")
    except Exception as e2:
        print(f"❌ Fallback conversion also failed: {e2}")
        raise RuntimeError("Unable to convert model to TFLite format")

# ---------------------------
# Save
# ---------------------------

with open(OUTPUT_FILE, "wb") as f:
    f.write(tflite_model)

print(
    f"✅ Saved {OUTPUT_FILE} ({CONVERT_MODE}, size: {os.path.getsize(OUTPUT_FILE)/1024:.2f} KB)")

# No cleanup needed for concrete function approach
