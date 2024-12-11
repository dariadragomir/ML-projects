import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import pandas as pd
import re
from sklearn.model_selection import train_test_split

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

class TextDataset(Dataset):
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

def get_dataloader(features, labels, batch_size=32, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = TextDataset(tokenizer, features, labels, max_length=max_length)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    return dataloader

train_features, test_features, train_labels, test_labels = train_test_split(
    features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

print(f"Training set size: {len(train_features)}")
print(f"Test set size: {len(test_features)}")

train_dataloader = get_dataloader(train_features, train_labels, batch_size=8, max_length=128)
test_dataloader = get_dataloader(test_features, test_labels, batch_size=8, max_length=128)

for batch in train_dataloader:
    print("Sample Train Batch:", batch)
    break

for batch in test_dataloader:
    print("Sample Test Batch:", batch)
    break
