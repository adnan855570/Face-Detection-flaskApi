import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from flask import Flask, request, jsonify
from src.anti_spoof_predict import Detection, AntiSpoofPredict
from src.data_io import transform as trans
from flask_cors import CORS

# --------------------------
# Config
# --------------------------
MODEL_DIR = "./resources/anti_spoof_models" 
DEVICE_ID = 0  # GPU id, or CPU fallback
app = Flask(__name__)
CORS(app)  # allow cross-origin requests

# --------------------------
# Load Models
# --------------------------
predictor = AntiSpoofPredict(device_id=DEVICE_ID)

model_paths = [
    os.path.join(MODEL_DIR, m)
    for m in os.listdir(MODEL_DIR)
    if m.endswith(".pth")
]

models = []
for path in model_paths:
    try:
        net, size = predictor._load_model(path)  # returns (model, input_size)
        net.eval()
        models.append((net, size))
        print(f"[INFO] Loaded model: {path}, input size: {size}")
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")

# face detector
detector = Detection()

# preprocessing
to_tensor = trans.Compose([trans.ToTensor()])


def preprocess(img, bbox, input_size):
    """Crop, resize, tensorize"""
    x, y, w, h = bbox
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, input_size)
    tensor = to_tensor(face)
    tensor = tensor.unsqueeze(0).to(predictor.device)
    return tensor


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # detect face
    try:
        bbox = detector.get_bbox(img)
    except Exception as e:
        return jsonify({"error": f"Face detection failed: {str(e)}"}), 500

    # run all models
    all_scores = []
    with torch.no_grad():
        for net, input_size in models:
            tensor = preprocess(img, bbox, input_size)
            result = net(tensor)
            result = F.softmax(result, dim=1).cpu().numpy().flatten()
            all_scores.append(result)

    if not all_scores:
        return jsonify({"error": "No models loaded"}), 500

    # average scores
    avg_scores = np.mean(all_scores, axis=0)
    label = int(np.argmax(avg_scores))
    confidence = float(avg_scores[label])

    return jsonify({
        "label": label,          # 0 = spoof, 1 = live
        "confidence": confidence,
        "raw_scores": avg_scores.tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
