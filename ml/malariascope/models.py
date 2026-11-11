"""Model zoo. Every model:

* takes raw 0-255 RGB pixels (scaling lives inside the graph -> no preprocessing mismatch at serve time),
* outputs a single **logit** for "Parasitized" (no sigmoid -> temperature scaling is exact),
* exposes a layer named ``feature_map`` (last spatial feature map) so Grad-CAM works on any of them.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers, regularizers

FEATURE_LAYER = "feature_map"


def _conv_bn_relu(x, filters: int, wd: float, name: str):
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(wd), name=f"{name}_conv"
    )(x)
    # momentum 0.9 (Keras default 0.99): running stats settle within the first epoch, so early-epoch
    # validation scores (and therefore early stopping) are trustworthy.
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_baseline_cnn(img_size: int, dropout: float = 0.3, weight_decay: float = 1e-4) -> keras.Model:
    """A compact VGG-style CNN (~1.2M params) trained from scratch — the honest baseline."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, f in enumerate((32, 64, 128, 256)):
        x = _conv_bn_relu(x, f, weight_decay, f"block{i + 1}a")
        x = _conv_bn_relu(x, f, weight_decay, f"block{i + 1}b")
        if i < 3:
            x = layers.MaxPooling2D(name=f"block{i + 1}_pool")(x)
            x = layers.SpatialDropout2D(0.1, name=f"block{i + 1}_drop")(x)
    x = layers.Activation("linear", name=FEATURE_LAYER)(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, name="logit", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def _build_transfer(base: keras.Model, img_size: int, dropout: float, name: str) -> keras.Model:
    base.trainable = False
    inputs = keras.Input((img_size, img_size, 3), name="image")
    # training=False keeps BatchNorm in inference mode even after unfreezing (standard fine-tuning recipe).
    x = base(inputs, training=False)
    x = layers.Activation("linear", name=FEATURE_LAYER)(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, name="logit", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=name)


def build_efficientnet_b0(img_size: int, dropout: float = 0.3, weights: str | None = "imagenet") -> keras.Model:
    # Keras' EfficientNet includes its own rescaling + normalisation and expects 0-255 input.
    base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    return _build_transfer(base, img_size, dropout, "efficientnet_b0")


def build_mobilenet_v3_small(img_size: int, dropout: float = 0.3, weights: str | None = "imagenet") -> keras.Model:
    base = keras.applications.MobileNetV3Small(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
        include_preprocessing=True,
        minimalistic=False,
    )
    return _build_transfer(base, img_size, dropout, "mobilenet_v3_small")


def build_model(name: str, img_size: int, dropout: float = 0.3, weights: str | None = "imagenet", weight_decay=1e-4):
    if name == "baseline_cnn":
        return build_baseline_cnn(img_size, dropout, weight_decay)
    if name == "efficientnet_b0":
        return build_efficientnet_b0(img_size, dropout, weights)
    if name == "mobilenet_v3_small":
        return build_mobilenet_v3_small(img_size, dropout, weights)
    raise ValueError(f"Unknown model {name!r}")


def backbone(model: keras.Model) -> keras.Model | None:
    """Return the nested pretrained backbone, or None for models trained from scratch."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


def unfreeze_top(model: keras.Model, n_layers: int) -> int:
    """Unfreeze the top ``n_layers`` of the backbone, keeping BatchNorm frozen. Returns #trainable layers."""
    base = backbone(model)
    if base is None:
        return 0
    base.trainable = True
    cutoff = max(0, len(base.layers) - n_layers)
    count = 0
    for i, layer in enumerate(base.layers):
        trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
        count += int(trainable)
    return count


def count_params(model: keras.Model) -> dict[str, int]:
    trainable = int(sum(keras.ops.size(w) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "trainable": trainable}
