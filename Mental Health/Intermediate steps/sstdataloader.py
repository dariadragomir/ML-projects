import nltk
import torch
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("merged_file.csv", on_bad_lines="skip")
print("Columns:", df.columns)
print("First 5 rows:", df.head())

features = df["post"]
labels = df["subreddit"]

stop_words = set(nltk.corpus.stopwords.words('english'))
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

encoded_labels, unique_labels = pd.factorize(labels)
label_mapping = dict(enumerate(unique_labels))
print("Label Mapping:", label_mapping)

train_features, val_features, train_labels, val_labels = train_test_split(
    features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

features_dict = {'train': train_features.tolist(), 'val': val_features.tolist()}
labels_dict = {'train': train_labels.tolist(), 'val': val_labels.tolist()}

class SSTdata(Dataset):
    def __init__(self, features, labels, sentiment_dict=None, max_length=100, mask_prob=0.99):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.features = features
        self.labels = labels
        self.sentiment_dict = sentiment_dict
        self.max_length = max_length
        self.mask_prob = mask_prob

        self.encodings = []
        for sentence in self.features:
            encoded_data = self.tokenizer.encode_plus(
                sentence, max_length=max_length, padding='max_length', truncation=True
            )
            self.encodings.append(encoded_data)

    def __getitem__(self, idx):
        item = {}
        temp_input_ids = self.encodings[idx]['input_ids']
        item['input_ids'] = torch.tensor(temp_input_ids)
        item['attention_mask'] = torch.tensor(self.encodings[idx]['attention_mask'])
        item['token_type_ids'] = torch.tensor(self.encodings[idx].get('token_type_ids', [0] * len(temp_input_ids))) 
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings)

def get_train_val_loaders_sstdata(features, labels, sentiment_dict, batch_size=64, epoch=0):
    dataset_train = SSTdata(features=features['train'], labels=labels['train'], sentiment_dict=sentiment_dict)
    dataset_val = SSTdata(features=features['val'], labels=labels['val'], sentiment_dict=sentiment_dict, mask_prob=0)

    train_dataloader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
    
    return train_dataloader, val_dataloader

sentiment_dict = None 
train_dataloader, val_dataloader = get_train_val_loaders_sstdata(features_dict, labels_dict, sentiment_dict)

def train_bert_model(train_dataloader, val_dataloader):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_labels))
    model.train() 

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3): 
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} completed. Evaluating on validation set...")
        evaluate_model(model, val_dataloader)

def evaluate_model(model, val_dataloader):
    model.eval()
    predictions, true_labels = [], []
    for batch in val_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=1).tolist())
        true_labels.extend(labels.tolist())

    acc = accuracy_score(true_labels, predictions)
    print("Validation Accuracy:", acc)
    print("Classification Report:\n", classification_report(true_labels, predictions))

train_bert_model(train_dataloader, val_dataloader)

vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_features).toarray()
X_test = vectorizer.transform(val_features).toarray()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, train_labels)
predictions = rf_model.predict(X_test)

acc = accuracy_score(val_labels, predictions)
print("Random Forest Accuracy:", acc)
print("Random Forest Classification Report:\n", classification_report(val_labels, predictions))
