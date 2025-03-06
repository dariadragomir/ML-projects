import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
import random
import math

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

pheromones = defaultdict(lambda: defaultdict(float))

def get_possible_pos(word):
    synsets = wn.synsets(word)
    pos_tags = set()
    for synset in synsets:
        pos_tags.add(synset.pos()) 
    return pos_tags if pos_tags else {'n'}

def initialize_pheromones(words):
    for word in words:
        pos_tags = get_possible_pos(word)
        for pos in pos_tags:
            pheromones[word][pos] = 1.0 

def select_pos(word):
    pos_tags = list(pheromones[word].keys())
    weights = [pheromones[word][pos] for pos in pos_tags]
    
    if sum(weights) == 0:
        return random.choice(pos_tags)
    
    return random.choices(pos_tags, weights=weights, k=1)[0]

def update_pheromones(sentence, tagged_sentence, evaporation=0.1, reward=2.0):
    for word, pos in tagged_sentence.items():
        for p in pheromones[word]:
            pheromones[word][p] *= (1 - evaporation)
        
        pheromones[word][pos] += reward

def aco_pos_tagger(sentence, iterations=100):
    words = nltk.word_tokenize(sentence)
    initialize_pheromones(words)
    best_tagging = {}
    best_score = -math.inf
    
    for _ in range(iterations):
        tagged_sentence = {word: select_pos(word) for word in words}
        score = sum(pheromones[word][tagged_sentence[word]] for word in words)
        
        if score > best_score:
            best_score = score
            best_tagging = tagged_sentence.copy()
        
        update_pheromones(words, tagged_sentence)
    
    return best_tagging

sentence = "Bank can guarantee deposits will be safe"
tagged_result = aco_pos_tagger(sentence)
