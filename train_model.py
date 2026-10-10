"""Train a MobileNetV2 plant-disease classifier and report REAL accuracy.

Dataset layout (e.g. PlantVillage from Kaggle):
    data/PlantVillage/<class_name>/*.jpg

Usage:
    python train_model.py --data_dir data/PlantVillage --epochs 5 --fine_tune_epochs 5

Outputs (in --out_dir, default ./models):
    plant_model.keras, plant_model.tflite, labels.json,
    metrics.json (test accuracy + per-class report), confusion_matrix.png
"""
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = 224
EXTS = {".jpg", ".jpeg", ".png"}


def list_files(data_dir):
    root = pathlib.Path(data_dir)
    classes, paths, labels = [], [], []
    for c in sorted(p for p in root.iterdir() if p.is_dir()):
        if c.name == root.name:
            continue  # skip a nested folder with the same name as the dataset root
        files = [f for f in c.iterdir() if f.suffix.lower() in EXTS]
        if not files:
            continue  # skip empty or nested folders
        for f in files:
            paths.append(str(f))
            labels.append(len(classes))
        classes.append(c.name)
    return classes, np.array(paths), np.array(labels)


def make_ds(paths, labels, batch, shuffle):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(n_classes):
    base = keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = keras.Input((IMG, IMG, 3))
    x = keras.layers.RandomFlip("horizontal_and_vertical")(inp)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomZoom(0.1)(x)
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--fine_tune_epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    classes, paths, labels = list_files(a.data_dir)
    print(f"{len(paths)} images, {len(classes)} classes")

    # Stratified 70 / 15 / 15 split, so test images are never seen in training
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=42)
    p_val, p_te, y_val, y_te = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    train = make_ds(p_tr, y_tr, a.batch, True)
    val = make_ds(p_val, y_val, a.batch, False)
    test = make_ds(p_te, y_te, a.batch, False)

    model, base = build_model(len(classes))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=a.epochs)

    if a.fine_tune_epochs > 0:  # unfreeze the top of MobileNetV2
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=a.fine_tune_epochs)

    # ---- honest evaluation on the held-out test set ----
    probs = model.predict(test)
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_te).mean())
    print(f"TEST ACCURACY: {acc * 100:.2f}%")

    report = classification_report(
        y_te, y_pred, labels=range(len(classes)), target_names=classes,
        output_dict=True, zero_division=0)
    (out / "metrics.json").write_text(json.dumps(
        {"test_accuracy": acc, "n_test_images": int(len(y_te)), "report": report}, indent=2))

    cm = confusion_matrix(y_te, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cm, cmap="Greens")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test accuracy {acc * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)

    # ---- save + export for mobile (TensorFlow Lite) ----
    model.save(out / "plant_model.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    (out / "plant_model.tflite").write_bytes(converter.convert())
    (out / "labels.json").write_text(json.dumps(classes, indent=2))
    print("Saved to", out.resolve())


if __name__ == "__main__":
    main()
