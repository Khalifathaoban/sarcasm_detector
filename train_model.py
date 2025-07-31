import os
import json
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm

# Constants
data_path = "sarcasm.json"
model_path = "sarcasm_model.pt"
download_url = "https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"

# Download dataset if not found
if not os.path.exists(data_path):
    print("🔄 Downloading dataset...")
    urllib.request.urlretrieve(download_url, data_path)
    print("✅ Download complete.")

# Load data
with open(data_path, 'r') as f:
    data = json.load(f)

# Preprocess text and labels
texts = [item["headline"] for item in data]
labels = [item["is_sarcastic"] for item in data]

# Build vocabulary
counter = Counter()
for text in texts:
    counter.update(text.lower().split())

vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(10000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# Encode text
def encode(text, vocab, max_len=40):
    tokens = text.lower().split()
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

encoded_texts = [encode(text, vocab) for text in texts]

# Create custom dataset
class SarcasmDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

# Dataset and DataLoader
dataset = SarcasmDataset(encoded_texts, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model definition
class SarcasmModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SarcasmModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

# Training setup
vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 128

model = SarcasmModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training starting...")
for epoch in range(5):
    total_loss = 0
    for X_batch, y_batch in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
