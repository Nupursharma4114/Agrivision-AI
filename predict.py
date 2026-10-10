"""Run the trained TFLite model on one image.

    python predict.py leaf.jpg --lang hi --question "इलाज कैसे करें?"
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image

from knowledge_base import answer
from quality import image_quality, reliability

try:  # lightweight runtime if installed, else full TensorFlow
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

IMG = 224


class Classifier:
    def __init__(self, model_dir="models"):
        d = pathlib.Path(model_dir)
        self.labels = json.loads((d / "labels.json").read_text())
        self.interp = Interpreter(model_path=str(d / "plant_model.tflite"))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict(self, img: Image.Image, top_k: int = 3):
        x = np.asarray(img.convert("RGB").resize((IMG, IMG)), dtype=np.float32)[None]
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out["index"])[0]
        idx = probs.argsort()[::-1][:top_k]
        return [(self.labels[i], float(probs[i])) for i in idx]


def analyze(clf: Classifier, img: Image.Image, question="", lang="en"):
    top = clf.predict(img)
    q = image_quality(img)
    rel = reliability(top[0][1], q["score"])
    return {
        "label": top[0][0],
        "confidence": round(top[0][1], 4),
        "alternatives": [{"label": l, "confidence": round(p, 4)} for l, p in top[1:]],
        "photo_quality": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()},
        "reliability": rel,
        "advice": answer(top[0][0], question, lang),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--lang", default="en", choices=["en", "hi"])
    ap.add_argument("--question", default="")
    a = ap.parse_args()
    result = analyze(Classifier(a.model_dir), Image.open(a.image), a.question, a.lang)
    print(json.dumps(result, indent=2, ensure_ascii=False))
