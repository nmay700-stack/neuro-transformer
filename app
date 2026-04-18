"""
app.py — NeuroTransformer Flask Backend
========================================
Run with:  python app.py
Opens at:  http://localhost:5000

Serves the EEG dashboard AND runs real Python inference.
No GPU required — CPU only.

Author: Mayuri Naveen | SRM KTR
"""

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import torch
import torch.nn.functional as F
import time, os, json

# ── import your model ──────────────────────────────────────
from model import NeuroTransformer, N_CHANNELS, N_SAMPLES, CH_NAMES, THRESHOLD

app = Flask(__name__, static_folder="static")

# ── load model once at startup ─────────────────────────────
print("Loading NeuroTransformer...")
MODEL = NeuroTransformer()
MODEL.eval()
print(f"  ✓ Model ready — {MODEL.count_parameters():,} parameters (CPU)")


# ── serve the dashboard ────────────────────────────────────
@app.route("/")
def index():
    """Serve the EEG dashboard HTML."""
    return send_from_directory(".", "index.html")


# ── inference endpoint ─────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "eeg": [[ch0_t0, ch0_t1, ...], [ch1_t0, ...], ...] }
          Shape: (22, 256) — 22 channels × 256 samples

    Returns: prediction, probabilities, attention matrix, latency
    """
    t0 = time.perf_counter()

    data = request.get_json()
    if not data or "eeg" not in data:
        return jsonify({"error": "Missing 'eeg' field"}), 400

    try:
        eeg = np.array(data["eeg"], dtype=np.float32)
        if eeg.shape != (N_CHANNELS, N_SAMPLES):
            # Try to reshape/pad if close
            if eeg.shape[0] == N_CHANNELS:
                if eeg.shape[1] < N_SAMPLES:
                    pad = np.zeros((N_CHANNELS, N_SAMPLES - eeg.shape[1]), dtype=np.float32)
                    eeg = np.concatenate([eeg, pad], axis=1)
                else:
                    eeg = eeg[:, :N_SAMPLES]
            else:
                return jsonify({
                    "error": f"Expected shape ({N_CHANNELS}, {N_SAMPLES}), got {eeg.shape}"
                }), 400
    except Exception as e:
        return jsonify({"error": f"EEG parse error: {str(e)}"}), 400

    # z-score per channel
    mean = eeg.mean(axis=1, keepdims=True)
    std  = eeg.std(axis=1,  keepdims=True) + 1e-8
    eeg  = (eeg - mean) / std

    # inference
    tensor = torch.from_numpy(eeg).unsqueeze(0)  # (1, 22, 256)
    with torch.no_grad():
        logits, attn = MODEL(tensor)
        probs = F.softmax(logits, dim=1)[0]

    p_ictal = probs[1].item()
    p_inter = probs[0].item()
    label   = "ictal" if p_ictal >= THRESHOLD else "interictal"
    latency = round((time.perf_counter() - t0) * 1000, 2)

    # attention matrix → list of lists
    attn_np = attn[0].cpu().numpy().tolist()

    return jsonify({
        "prediction":    label,
        "probabilities": {"ictal": p_ictal, "interictal": p_inter},
        "attentionLayer1": attn_np,
        "latencyMs":     latency,
        "alert":         p_ictal >= THRESHOLD,
        "threshold":     THRESHOLD,
        "channelNames":  CH_NAMES,
    })


# ── batch inference endpoint ───────────────────────────────
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    POST /predict/batch
    Body: { "windows": [ [[22×256]], [[22×256]], ... ] }
    Returns predictions for all windows.
    """
    data = request.get_json()
    if not data or "windows" not in data:
        return jsonify({"error": "Missing 'windows' field"}), 400

    results = []
    for i, win in enumerate(data["windows"]):
        try:
            eeg = np.array(win, dtype=np.float32)
            if eeg.shape[0] != N_CHANNELS:
                results.append({"window": i, "error": "wrong channel count"})
                continue
            if eeg.shape[1] < N_SAMPLES:
                eeg = np.pad(eeg, ((0,0),(0, N_SAMPLES-eeg.shape[1])))
            eeg = eeg[:, :N_SAMPLES]

            mean = eeg.mean(axis=1, keepdims=True)
            std  = eeg.std(axis=1,  keepdims=True) + 1e-8
            eeg  = (eeg - mean) / std

            tensor = torch.from_numpy(eeg).unsqueeze(0)
            with torch.no_grad():
                logits, _ = MODEL(tensor)
                probs = F.softmax(logits, dim=1)[0]

            p_ictal = probs[1].item()
            results.append({
                "window": i,
                "prediction": "ictal" if p_ictal >= THRESHOLD else "interictal",
                "p_ictal": round(p_ictal, 4),
                "alert": p_ictal >= THRESHOLD,
            })
        except Exception as e:
            results.append({"window": i, "error": str(e)})

    return jsonify({"results": results, "count": len(results)})


# ── model info endpoint ────────────────────────────────────
@app.route("/info")
def info():
    return jsonify({
        "model":       "NeuroTransformer v1.0",
        "parameters":  MODEL.count_parameters(),
        "channels":    N_CHANNELS,
        "sampleRate":  256,
        "windowSec":   1,
        "threshold":   THRESHOLD,
        "dataset":     "CHB-MIT Subject 01",
        "author":      "Mayuri Naveen, SRM KTR",
        "github":      "https://github.com/nmay700-stack",
    })


# ── health check ───────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})


# ── run ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  NeuroTransformer — Flask Backend")
    print("  Dashboard → http://localhost:5000")
    print("  API docs  → http://localhost:5000/info")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
