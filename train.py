import os
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.saving import register_keras_serializable

# ---------------------------
# Custom Triplet Semi-Hard Loss
# ---------------------------


class TripletSemiHardLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, name="triplet_semihard_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        # Pairwise distances
        pairwise_dist = tf.norm(
            tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0), axis=2
        )

        adjacency = tf.equal(tf.expand_dims(y_true, 0),
                             tf.expand_dims(y_true, 1))
        adjacency_not = tf.logical_not(adjacency)

        mask = tf.logical_not(tf.eye(tf.shape(y_true)[0], dtype=tf.bool))

        hardest_positive_dist = tf.reduce_max(
            tf.where(adjacency & mask, pairwise_dist,
                     tf.zeros_like(pairwise_dist)),
            axis=1,
        )
        hardest_negative_dist = tf.reduce_min(
            tf.where(adjacency_not, pairwise_dist,
                     tf.ones_like(pairwise_dist) * 1e12),
            axis=1,
        )

        loss = tf.maximum(
            hardest_positive_dist - hardest_negative_dist + self.margin, 0.0
        )
        return tf.reduce_mean(loss)

# ---------------------------
# Custom L2Normalization Layer
# ---------------------------


@register_keras_serializable(package="Custom", name="L2Normalization")
class L2Normalization(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# ---------------------------
# Define MobileFaceNet
# ---------------------------


def depthwise_conv_block(inputs, pointwise_filters, strides=(1, 1)):
    x = layers.DepthwiseConv2D(3, strides=strides, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(pointwise_filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    return x


def MobileFaceNet(input_shape=(112, 112, 3), embedding_size=128):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    x = depthwise_conv_block(x, 64)
    x = depthwise_conv_block(x, 128, strides=(2, 2))
    x = depthwise_conv_block(x, 128)
    x = depthwise_conv_block(x, 128)
    x = depthwise_conv_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_size)(x)

    outputs = L2Normalization()(x)

    return models.Model(inputs, outputs, name="MobileFaceNet")

# ---------------------------
# Dataset Loader
# ---------------------------


def get_dataset():
    faces_path = "faces/lfw-deepfunneled/lfw-deepfunneled"
    if os.path.exists(faces_path):
        print(f"📥 Using dataset from {faces_path}")
        ds = tf.keras.utils.image_dataset_from_directory(
            faces_path,
            image_size=(112, 112),
            batch_size=16
        ).map(lambda x, y: (x / 255.0, tf.cast(y, tf.int32)))
        return ds
    else:
        print("⚠️ Dataset not found, using synthetic data for testing...")
        x = tf.random.uniform((500, 112, 112, 3))
        y = tf.random.uniform((500,), maxval=10, dtype=tf.int32)
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(16)

# ---------------------------
# Train
# ---------------------------


train_ds = get_dataset()
model = MobileFaceNet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=TripletSemiHardLoss(margin=1.0)
)

print("🚀 Training MobileFaceNet...")
# model.fit(train_ds.take(10), epochs=2)  # quick test run
model.fit(train_ds, epochs=70)

model.save("mobilefacenet.keras")
print("✅ Saved model: mobilefacenet.keras")
