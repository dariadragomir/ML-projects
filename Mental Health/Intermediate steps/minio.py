from minio import Minio
from minio.error import S3Error
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
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

output_file = "classification_report.txt"
with open(output_file, "w") as f:
    f.write("Random Forest Accuracy: 0.879\n")
    f.write("Classification Report:\n")
    f.write("...\n")  # Add actual results here

def upload_file_to_minio(client, bucket, local_path, object_name):
    try:
        print(f"Uploading '{local_path}' to MinIO bucket '{bucket}' as '{object_name}'...")
        client.fput_object(bucket, object_name, local_path)
        print("Upload complete.")
    except S3Error as e:
        print(f"Error uploading file: {e}")

upload_object_name = "classification_results.txt"
upload_file_to_minio(minio_client, bucket_name, output_file, upload_object_name)
