"""
Model definitions for Swara keyword spotting / speech recognition.
Designed for low-latency, low-memory edge deployments (TFLite / Microcontrollers).
"""

from typing import Tuple
from config import DEFAULT_INPUT_SHAPE, DEFAULT_NUM_CLASSES, DEFAULT_NUM_FILTERS


def build_dscnn_model(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    num_filters: int = DEFAULT_NUM_FILTERS,
    bn_momentum: float = 0.80,
):
    """
    Build a Depthwise Separable Convolutional Neural Network (DS-CNN).
    Optimized for Swara Frozen Audio Spec V0 (49 frames x 10 MFCCs).

    Architecture Notes:
    - BatchNorm momentum is set to 0.80 to ensure running mean/variance statistics
      converge stably on small-to-medium edge audio datasets.
    - Classification Head has two distinct layers:
      1. 'logits': Dense layer with activation=None (TFLite Op: FULLY_CONNECTED)
      2. 'output': Softmax layer (TFLite Op: SOFTMAX)
      This represents a single mathematical Softmax mapping. Downstream inference
      consumers (live_mic_test.py, TFLM C runtime) receive normalized probabilities
      directly and MUST NOT apply an additional softmax.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.Input(shape=input_shape),
            # Standard 2D Conv layer (strided over time and frequency)
            layers.Conv2D(num_filters, (5, 3), strides=(2, 1), padding="same", use_bias=False),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.ReLU(),
            # Depthwise Separable Conv Block 1
            layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.ReLU(),
            layers.Conv2D(num_filters, (1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.ReLU(),
            # Depthwise Separable Conv Block 2
            layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.ReLU(),
            layers.Conv2D(num_filters, (1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.ReLU(),
            # Pooling & Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation=None, name="logits"),
            layers.Softmax(name="output"),
        ])
        return model
    except ImportError:
        print("TensorFlow not installed. Please install tensorflow to build the model.")
        return None


if __name__ == "__main__":
    model = build_dscnn_model()
    if model:
        model.summary()
