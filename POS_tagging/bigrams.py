import re
import glob
from collections import defaultdict

def parse_pos_files(file_pattern):
    all_sentences = []
    exclude_tags = [',', '.', '``', "''", '(', ')', ':', '|','#', '$', ";"]  # Tags to exclude
    
    for file_path in glob.glob(file_pattern):
        with open(file_path, 'r', encoding='utf-8') as file:
            current_sentence = []
            
            for line in file:
                stripped_line = line.strip()
                
                if stripped_line == './.':
                    if current_sentence:
                        all_sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                if not stripped_line:
                    continue
                
                elements = re.findall(r'\[.*?\]|\S+', stripped_line)
                for elem in elements:
                    if elem.startswith('[') and elem.endswith(']'):
                        content = elem[1:-1].strip()
                        if content:
                            for token in content.split():
                                parts = token.rsplit('/', 1)
                                if len(parts) == 2:
                                    tag = parts[1]
                                    if tag not in exclude_tags and '|' not in tag:
                                        current_sentence.append(tag)
                    else:
                        parts = elem.rsplit('/', 1)
                        if len(parts) == 2:
                            tag = parts[1]
                            if tag not in exclude_tags and '|' not in tag:
                                current_sentence.append(tag)
            
            if current_sentence:
                all_sentences.append(current_sentence)
    
    return all_sentences

def compute_bigram_probabilities(sentences):
    bigram_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    for sentence in sentences:
        for a, b in zip(sentence[:-1], sentence[1:]):
            bigram_counts[(a, b)] += 1
            tag_counts[a] += 1
    
    probabilities = {}
    for (a, b), count in bigram_counts.items():
        if tag_counts[a] > 0:
            probabilities[(a, b)] = count / tag_counts[a]
    
    return probabilities

file_pattern = 'treebank/tagged/*.pos'
sentences = parse_pos_files(file_pattern)
bigram_probs = compute_bigram_probabilities(sentences)

print("Bigram Probabilities:")
for (a, b), prob in sorted(bigram_probs.items(), key=lambda x: (-x[1], x[0])):
    print(f"P({b}|{a}) = {prob:.4f}")

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

def parse_text(text):
    sentences = sent_tokenize(text)
    return [word_tokenize(sent) for sent in sentences]

input_text = """
Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.
Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group!
What's the state-of-the-art in 2023?
"""

parsed = parse_text(input_text)

for i, sentence_words in enumerate(parsed, 1):
    print(f"Sentence {i}:")
    print(f"Words: {sentence_words}\n")
