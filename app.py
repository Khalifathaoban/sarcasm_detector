import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import re

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define model architecture
class SarcasmModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SarcasmModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# Sample vocabulary and preprocessing
vocab = {"i": 1, "love": 2, "this": 3, "movie": 4, "so": 5, "much": 6, "not": 7, "funny": 8, "great": 9, "job": 10}
vocab_size = 10002
embed_dim = 64
hidden_dim = 128
output_dim = 2

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    indexed = [vocab.get(token, 0) for token in tokens]
    return torch.tensor(indexed).unsqueeze(0)

# Load the trained model
model = SarcasmModel(vocab_size, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("sarcasm_model.pth", map_location=torch.device("cpu")))
model.eval()

# Routes
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")
    print("Received data:", data)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        input_tensor = preprocess(text)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        prediction_label = "Sarcastic" if predicted_class == 1 else "Not Sarcastic"

        # 👇 Add these debug print statements
        print(f"Text received: {text}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence}")
        print(f"Full probabilities: {probabilities.numpy()}")

        return jsonify({
            "text": text,
            "prediction": prediction_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")  # 👈 Also helpful
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

sample_text = "I absolutely love waiting in long lines for hours."
input_tensor = preprocess(sample_text)

with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    print("Prediction:", prediction)
    print("Probabilities:", probs)

# Run the app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

# Test after loading model
