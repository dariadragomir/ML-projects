import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

df = pd.read_csv("meta.csv")
df = df[df['label_T'].notnull()]
texts = df['text'].tolist()
labels = df['label_T'].astype(float).tolist()


embedding_models = {
    "sbert-chinese-nli": "uer/sbert-base-chinese-nli",
    "text2vec": "shibing624/text2vec-base-chinese",
    "sbert-chinese-nli": "uer/sbert-base-chinese-nli",
    "text2vec-chinese": "shibing624/text2vec-base-chinese",
    "chinese-roberta": "hfl/chinese-roberta-wwm-ext",
    "mengzi-bert": "Langboat/mengzi-bert-base",
    "multilingual-MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

def evaluate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).flatten()

    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "F1 Score": f1_score(y_true_bin, y_pred_bin),
        "Precision": precision_score(y_true_bin, y_pred_bin),
        "Recall": recall_score(y_true_bin, y_pred_bin)
    }

def train_model(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Regressor(input_dim=embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    return evaluate(y_test, predictions)

results = {}

for name, model_id in embedding_models.items():
    print(f"\n Testing model: {name}")
    embedder = SentenceTransformer(model_id)
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    metrics = train_model(np.array(embeddings), labels)
    results[name] = metrics

print("\n Final Comparison:")
for model_name, metric in results.items():
    print(f"\nModel: {model_name}")
    for k, v in metric.items():
        print(f"{k}: {v:.4f}")

'''

Model: sbert-chinese-nli
MAE: 0.3894
MSE: 0.2202
R2: 0.4174
Accuracy: 0.7866
F1 Score: 0.7235
Precision: 0.6949
Recall: 0.7546

Model: text2vec
MAE: 0.4391
MSE: 0.2703
R2: 0.2849
Accuracy: 0.7730
F1 Score: 0.6855
Precision: 0.7032
Recall: 0.6687

Model: text2vec-chinese
MAE: 0.4343
MSE: 0.2645
R2: 0.3003
Accuracy: 0.7537
F1 Score: 0.6766
Precision: 0.6580
Recall: 0.6963

Model: chinese-roberta
MAE: 0.4810
MSE: 0.3210
R2: 0.1508
Accuracy: 0.7367
F1 Score: 0.5827
Precision: 0.7043
Recall: 0.4969

Model: mengzi-bert
MAE: 0.4873
MSE: 0.3298
R2: 0.1274
Accuracy: 0.7208
F1 Score: 0.4459
Precision: 0.8390
Recall: 0.3037

Model: multilingual-MiniLM
MAE: 0.5092
MSE: 0.3591
R2: 0.0498
Accuracy: 0.6470
F1 Score: 0.1289
Precision: 0.7419
Recall: 0.0706

'''
