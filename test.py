import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter("ace_embedding.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def get_embedding(img_path):
    img = Image.open(img_path).resize((112, 112))
    x = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]


def is_match(embedding1, embedding2, threshold=1.0):
    """
    Check if two face embeddings match based on cosine similarity.

    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector
        threshold: Similarity threshold (default: 1.0)
                  Higher values = more strict matching

    Returns:
        bool: True if embeddings match (similarity >= threshold), False otherwise
    """
    cosine_similarity = np.dot(embedding1, embedding2) / \
        (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cosine_similarity >= threshold


# ---- Test with sample images ----
try:
    emb1 = get_embedding("faces_test/pic2.png")
    emb2 = get_embedding("faces_test/pic3.jpg")

    cosine = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print("Cosine similarity:", cosine)

    # Test the is_match method
    match_result = is_match(emb1, emb2)
    print("Faces match:", match_result)

    # Test with different thresholds
    strict_match = is_match(emb1, emb2, threshold=1.0)
    loose_match = is_match(emb1, emb2, threshold=0.4)
    print("Strict match (1.0):", strict_match)
    print("Loose match (0.4):", loose_match)

except Exception as e:
    print("⚠️ Error running inference:", e)
    print("Tip: place two test images named image1.png and image2.png in your project folder")
