import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# Define the model class again
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

# Load model
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

model = SarcasmModel(vocab_size, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("sarcasm_model.pth", map_location=torch.device("cpu")))
model.eval()

# Streamlit UI
st.set_page_config(page_title="Sarcasm Detector")
st.title("🤖 Sarcasm Detection")
st.write("Type a sentence and see if it detects sarcasm.")

user_input = st.text_area("Enter your sentence")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_tensor = preprocess(user_input)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

        label = "Sarcastic 😏" if predicted_class == 1 else "Not Sarcastic 🙂"
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {round(confidence * 100, 2)}%")
