from minio import Minio
from minio.error import S3Error
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import re

minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "datasets"
object_name = "merged_file.csv"
download_path = "temp_merged_file.csv"

def download_file_from_minio(client, bucket, object_name, local_path):
    try:
        print(f"Downloading '{object_name}' from MinIO bucket '{bucket}'...")
        client.fget_object(bucket, object_name, local_path)
        print(f"File downloaded to '{local_path}'.")
    except S3Error as e:
        print(f"Error downloading file: {e}")

if not os.path.exists(download_path):
    download_file_from_minio(minio_client, bucket_name, object_name, download_path)

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv(download_path, on_bad_lines="skip")
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

vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_features).toarray()
X_test = vectorizer.transform(test_features).toarray()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, train_labels)
predictions = rf_model.predict(X_test)

rf_acc = accuracy_score(test_labels, predictions)
rf_report = classification_report(test_labels, predictions)

print("Random Forest Accuracy:", rf_acc)
print("Random Forest Classification Report:\n", rf_report)

output_file = "classification_report_rf.txt"
with open(output_file, "w") as f:
    f.write(f"Random Forest Accuracy: {rf_acc}\n")
    f.write("Classification Report:\n")
    f.write(rf_report)

def upload_file_to_minio(client, bucket, local_path, object_name):
    try:
        print(f"Uploading '{local_path}' to MinIO bucket '{bucket}' as '{object_name}'...")
        client.fput_object(bucket, object_name, local_path)
        print("Upload complete.")
    except S3Error as e:
        print(f"Error uploading file: {e}")

upload_object_name = "classification_results_rf.txt"
upload_file_to_minio(minio_client, bucket_name, output_file, upload_object_name)
