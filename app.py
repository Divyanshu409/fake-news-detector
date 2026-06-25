from flask import Flask, render_template, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import os
import json

app = Flask(__name__)

MODEL_PATH = "models/fake_news_model"
tokenizer = None
model = None
device = torch.device("cpu")


def load_model():
    global tokenizer, model
    if os.path.exists(MODEL_PATH):
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
    else:
        print("No saved model found. Please run train.py first.")


def get_top_keywords(text, tokenizer, n=10):
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "this", "that",
        "these", "those", "it", "its", "as", "not", "no", "so", "if",
        "about", "up", "out", "than", "then", "he", "she", "they",
        "we", "you", "i", "his", "her", "their", "our", "your", "my",
        "said", "also", "after", "before", "into", "more", "there"
    }
    words = text.lower().split()
    keywords = [w.strip(".,!?\"'()[]{}:;") for w in words
                if w.strip(".,!?\"'()[]{}:;") not in stopwords
                and len(w.strip(".,!?\"'()[]{}:;")) > 3]
    seen = set()
    unique_kw = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_kw.append(kw)
    return unique_kw[:n]


def highlight_text(text, keywords):
    import re
    highlighted = text
    for kw in keywords:
        pattern = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
        highlighted = pattern.sub(r'<mark>\1</mark>', highlighted)
    return highlighted


@app.route("/")
def index():
    model_ready = model is not None
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please run train.py first."}), 503

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text or len(text) < 20:
        return jsonify({"error": "Please enter a news article (at least 20 characters)."}), 400

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    predicted_class = int(np.argmax(probs))
    confidence = round(float(max(probs)) * 100, 2)
    label = "REAL" if predicted_class == 1 else "FAKE"

    keywords = get_top_keywords(text, tokenizer, n=12)
    highlighted = highlight_text(text, keywords)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "highlighted_text": highlighted,
        "keywords": keywords,
        "fake_prob": round(probs[0] * 100, 2),
        "real_prob": round(probs[1] * 100, 2)
    })


@app.route("/metrics")
def metrics():
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No metrics found. Run train.py to generate them."}), 404


if __name__ == "__main__":
    load_model()
    app.run(debug=False, port=5000)