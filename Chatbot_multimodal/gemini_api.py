import os
import json
import csv
import requests
from constants import PREDICTIONS_DIR, META_CSV, GEMINI_API_KEY, GEMINI_URL

def get_meta_lines(csv_path):
    meta_data = {}
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = f"{row['video_id']}_{int(row['clip_id']):04d}"
            meta_data[key] = row['text']
    return meta_data

def get_last_sentiment_mean(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data or 'mean_emotions' not in data[-1]:
        return None

    return data[-1]['mean_emotions']

def build_prompt(sentiments, chinese_line):
    return f"""
You are a Chinese translator that analyses some videos from a Chinese drama. 
From each video, an array of sentiments has been extracted (the percent of anger, disgust, fear, happiness, sadness, surprise, and neutral if there were any other emotions). 
I already analysed the video and I will give you the result, as well as one Chinese line that was said by one of the actors. 
The sentiments are {json.dumps(sentiments, ensure_ascii=False)}
The line is "{chinese_line}".
Tell me if the overall sentiment is Positive, Negative or Neutral.
Also say a label between [-1, 1] for the main emotion in the line, where -1 means negative and 1 positive. 
Answer short.
"""

def query_gemini(prompt, api_key):
    headers = {
        'Content-Type': 'application/json',
    }
    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    params = {
        'key': api_key
    }

    response = requests.post(GEMINI_URL, headers=headers, params=params, json=body)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        print("Error querying Gemini:", response.text)
        return None

def main():
    meta_lines = get_meta_lines(META_CSV)

    for filename in os.listdir(PREDICTIONS_DIR):
        if filename.endswith('_predictions.json'):
            path = os.path.join(PREDICTIONS_DIR, filename)
            sentiment_means = get_last_sentiment_mean(path)

            if not sentiment_means:
                continue

            key = filename.replace('_predictions.json', '')
            if key in meta_lines:
                chinese_line = meta_lines[key]
                prompt = build_prompt(sentiment_means, chinese_line)
                print(f"\n--- {key} ---")
                response = query_gemini(prompt, GEMINI_API_KEY)
                print("Gemini response:", response)

if __name__ == "__main__":
    main()
