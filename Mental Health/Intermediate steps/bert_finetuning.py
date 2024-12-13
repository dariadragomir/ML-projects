import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import pandas as pd
import re
import numpy as np
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("merged_file.csv", on_bad_lines="skip")
print("Columns:", df.columns)
print("First 5 rows:", df.head())

features = df["post"]
labels = df["subreddit"]

encoded_labels, unique_labels = pd.factorize(labels)
label_mapping = dict(enumerate(unique_labels))
print("Label Mapping:", label_mapping)

stop_words = set(stopwords.words('english'))
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F" 
    u"\U0001F300-\U0001F5FF" 
    u"\U0001F680-\U0001F6FF" 
    u"\U0001F1E0-\U0001F1FF" 
    u"\U00002500-\U00002BEF" 
    u"\U00002702-\U000027B0"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

def preprocess_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens]
    tokens = [emoji_pattern.sub(r'', word) for word in tokens]
    return ' '.join([token for token in tokens if token])

features = features.apply(preprocess_text)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# tokenizer + BERT
class BERTDataset(Dataset):
    def __init__(self, tokenizer, features, labels, max_length=128):
        self.tokenizer = tokenizer
        self.features = features
        self.labels = labels
        self.max_length = max_length
        self.encodings = self.tokenizer(
            self.features.tolist(),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.labels)

def get_bert_dataloader(features, labels, batch_size=8, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = BERTDataset(tokenizer, features, labels, max_length=max_length)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    return dataloader

def train_bert_model(train_dataloader, test_dataloader, epochs=3, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_labels))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Average training loss: {total_loss / len(train_dataloader)}")

    model.eval()
    predictions, true_labels = [], []
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=1).tolist())
        true_labels.extend(labels.tolist())

    acc = accuracy_score(true_labels, predictions)
    print("BERT Accuracy:", acc)
    print("BERT Classification Report:\n", classification_report(true_labels, predictions))

train_dataloader = get_bert_dataloader(train_features, train_labels, batch_size=16, max_length=128)
test_dataloader = get_bert_dataloader(test_features, test_labels, batch_size=16, max_length=128)

train_bert_model(train_dataloader, test_dataloader, epochs=5, learning_rate=2e-5)

# countVectorizer + Random Forest
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_features).toarray()
X_test = vectorizer.transform(test_features).toarray()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, train_labels)
predictions = rf_model.predict(X_test)

acc = accuracy_score(test_labels, predictions)
print("Random Forest Accuracy:", acc)
print("Random Forest Classification Report:\n", classification_report(test_labels, predictions))

'''
Epoch 1/5
Average training loss: 0.4296777323386941
Epoch 2/5
Average training loss: 0.2170338969252908
Epoch 3/5
Average training loss: 0.12874838068524905
Epoch 4/5
Average training loss: 0.07230212055774889
Epoch 5/5
Average training loss: 0.04408032156672125

BERT Accuracy: 0.9055139186295503
BERT Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.94      0.94       820
           1       0.95      0.93      0.94      1022
           2       0.86      0.86      0.86       998
           3       0.87      0.89      0.88       896

    accuracy                           0.91      3736
   macro avg       0.91      0.91      0.91      3736
weighted avg       0.91      0.91      0.91      3736


Random Forest Accuracy: 0.8792826552462527
Random Forest Classification Report:
               precision    recall  f1-score   support

           0       0.91      0.89      0.90       820
           1       0.88      0.92      0.90      1022
           2       0.82      0.86      0.84       998
           3       0.93      0.84      0.88       896

    accuracy                           0.88      3736
   macro avg       0.88      0.88      0.88      3736
weighted avg       0.88      0.88      0.88      3736

'''
