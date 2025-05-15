import pandas as pd
from transformers import pipeline
import json
from tqdm import tqdm

df = pd.read_csv("meta.csv")
df = df[df['text'].notnull()]
texts = df['text'].tolist()

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

def predict_emotion_zero_shot(text):
    try:
        result = classifier(text, candidate_labels=emotion_labels)
        scores = dict(zip(result["labels"], result["scores"]))

        return {label: float(scores.get(label, 0.0)) for label in emotion_labels}
    except Exception as e:
        print(f"Error analyzing text: {text[:30]}... => {e}")
        return {label: 0.0 for label in emotion_labels}

emotion_predictions = []

print("Processing Chinese texts for emotion classification...")
for text in tqdm(texts, desc="Classifying"):
    emotion_predictions.append(predict_emotion_zero_shot(text))

with open("text_emotion_predictions.json", "w", encoding="utf-8") as f:
    json.dump(emotion_predictions, f, ensure_ascii=False, indent=2)

emotion_df = pd.DataFrame(emotion_predictions)
df_output = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)
df_output.to_csv("text_emotion_output.csv", index=False, encoding="utf-8-sig")

