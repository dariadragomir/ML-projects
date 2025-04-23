# AutoModel+weighted pooling

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm

MODEL_NAME = "uer/sbert-base-chinese-nli"
BATCH_SIZE = 16
MAX_LEN = 64
EPOCHS = 10
LR = 2e-5

meta = pd.read_csv("meta.csv")
texts = meta["text"].astype(str).tolist()
labels = meta["label_T"].astype(float).tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ChineseSentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class WeightedPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, hidden_states, attention_mask):
        weights = self.attention(hidden_states).squeeze(-1)
        weights = weights.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)
        return torch.sum(weights * hidden_states, dim=1)

class SentimentRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        print(self.encoder)
        hidden_size = self.encoder.config.hidden_size
        self.pooling = WeightedPooling(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooling(outputs.last_hidden_state, attention_mask)
        x = self.dropout(pooled)
        return self.regressor(x).squeeze(-1)

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = ChineseSentimentDataset(train_texts, train_labels)
test_dataset = ChineseSentimentDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = SentimentRegressor().to(device)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        outputs = model(input_ids, attention_mask).cpu().numpy()
        preds.extend(outputs)
        trues.extend(labels)

mae = mean_absolute_error(trues, preds)
mse = mean_squared_error(trues, preds)
r2 = r2_score(trues, preds)


binary_preds = [1 if p > 0 else 0 for p in preds]
binary_trues = [1 if t > 0 else 0 for t in trues]
accuracy = accuracy_score(binary_trues, binary_preds)
f1 = f1_score(binary_trues, binary_preds)
precision = precision_score(binary_trues, binary_preds)
recall = recall_score(binary_trues, binary_preds)

print("\nFinal Evaluation Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


'''
Final Evaluation Metrics:
MAE: 0.2659
MSE: 0.1269
R2: 0.6644
Accuracy: 0.8263
F1 Score: 0.7895
Precision: 0.7157
Recall: 0.8804
'''
