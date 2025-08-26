from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize Flask
app = Flask(__name__)

# Load the model once at startup
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

def compute_similarity(text1, text2):
    # Encode sentences into embeddings
    emb1 = model.encode(text1, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = model.encode(text2, convert_to_tensor=True, normalize_embeddings=True)

    # Cosine similarity
    cos_sim = util.cos_sim(emb1, emb2).item()
    # Scale to [0,1]
    return (cos_sim + 1) / 2

@app.route("/similarity", methods=["POST"])
def similarity():
    try:
        data = request.get_json()
        text1 = data.get("text1", "")
        text2 = data.get("text2", "")

        if not text1 or not text2:
            return jsonify({"error": "Both 'text1' and 'text2' must be provided"}), 400

        score = compute_similarity(text1, text2)
        return jsonify({"similarity score": round(float(score), 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Run on 0.0.0.0 for cloud deployment
    app.run(host="0.0.0.0", port=8080, debug=False)
