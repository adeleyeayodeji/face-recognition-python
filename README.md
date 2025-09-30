# Face Recognition with MobileFaceNet

A lightweight face recognition system using MobileFaceNet architecture, optimized for mobile deployment with TensorFlow Lite.

## 🚀 Features

- **MobileFaceNet Architecture**: Lightweight CNN optimized for mobile devices
- **Triplet Loss Training**: Uses semi-hard triplet loss for better face embeddings
- **TensorFlow Lite Support**: Convert trained models to TFLite for mobile deployment
- **Face Embedding Extraction**: Generate 128-dimensional face embeddings
- **Cosine Similarity Matching**: Compare faces using cosine similarity

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x (with Apple Silicon support)
- OpenCV
- Pillow
- NumPy
- Pandas
- Scikit-learn

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd face-recognition-python
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
face-recognition-python/
├── train.py              # Model training script
├── convert.py            # TFLite conversion script
├── test.py               # Model testing and inference
├── mobilefacenet.keras   # Trained Keras model
├── ace_embedding.tflite  # TFLite model for mobile deployment
├── faces/                # Training dataset (LFW)
│   └── lfw-deepfunneled/
├── faces_test/           # Test images
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 Usage

### 1. Training the Model

Train the MobileFaceNet model on the LFW dataset:

```bash
python train.py
```

**Note**: The training script will:

- Load the LFW dataset from `faces/lfw-deepfunneled/`
- Train for 20 epochs using triplet semi-hard loss
- Save the trained model as `mobilefacenet.keras`

### 2. Converting to TensorFlow Lite

Convert the trained Keras model to TFLite format for mobile deployment:

```bash
python convert.py
```

This will:

- Load the trained model
- Convert to TFLite format with fp16 quantization
- Save as `ace_embedding.tflite` (optimized for mobile)

### 3. Testing Face Recognition

Test the model with sample images:

```bash
python test.py
```

**Before running tests**:

1. Place test images in the `faces_test/` directory
2. Update the image paths in `test.py` (lines 41-42)

Example test images:

- `faces_test/image1.png`
- `faces_test/image2.png`

## 🔧 Configuration

### Model Parameters

In `train.py`, you can modify:

- `embedding_size`: Size of face embeddings (default: 128)
- `input_shape`: Input image dimensions (default: 112x112x3)
- `epochs`: Number of training epochs (default: 20)
- `batch_size`: Training batch size (default: 16)

### Conversion Settings

In `convert.py`, you can adjust:

- `CONVERT_MODE`: "fp16" for mobile, "fp32" for server
- `OUTPUT_FILE`: Output TFLite filename

### Similarity Threshold

In `test.py`, adjust the matching threshold:

- `threshold=1.0`: Strict matching (higher precision)
- `threshold=0.4`: Loose matching (higher recall)

## 📊 Model Architecture

The MobileFaceNet architecture includes:

- **Input**: 112x112x3 RGB images
- **Backbone**: Depthwise separable convolutions
- **Output**: 128-dimensional L2-normalized embeddings
- **Loss**: Triplet semi-hard loss for training

## 🧪 Testing

### Face Embedding Extraction

```python
from test import get_embedding

# Extract face embedding
embedding = get_embedding("path/to/face.jpg")
print(f"Embedding shape: {embedding.shape}")  # (128,)
```

### Face Matching

```python
from test import is_match

# Compare two face embeddings
match = is_match(embedding1, embedding2, threshold=0.8)
print(f"Faces match: {match}")
```

## 🚨 Troubleshooting

### Common Issues

1. **"Model not found" error**:

   - Ensure `mobilefacenet.keras` exists
   - Run `train.py` first to create the model

2. **TFLite conversion errors**:

   - The conversion script includes fallback methods
   - Try different `CONVERT_MODE` settings

3. **Dataset not found**:

   - Place LFW dataset in `faces/lfw-deepfunneled/`
   - Or the script will use synthetic data for testing

4. **Memory issues during training**:
   - Reduce `batch_size` in `train.py`
   - Use fewer training epochs

### Performance Tips

- **Training**: Use GPU acceleration if available
- **Inference**: TFLite model is optimized for mobile devices
- **Accuracy**: Adjust similarity threshold based on your use case

## 📈 Performance

- **Model Size**: ~186 KB (TFLite fp16)
- **Input Size**: 112x112x3
- **Output Size**: 128-dimensional embeddings
- **Inference Time**: <10ms on modern mobile devices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LFW (Labeled Faces in the Wild) dataset
- MobileFaceNet paper and architecture
- TensorFlow and TensorFlow Lite teams
