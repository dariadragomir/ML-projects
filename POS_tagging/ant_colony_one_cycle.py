import os
import json
import random
from collections import defaultdict

class ACOPOSTagger:
    def __init__(self, pheromone_file="pheromones.json"):
        self.pheromone_file = pheromone_file
        self.pheromones = defaultdict(lambda: defaultdict(float))
        self.load_pheromones()
    
    def load_pheromones(self):
        if os.path.exists(self.pheromone_file):
            with open(self.pheromone_file, "r", encoding="utf-8") as f:
                self.pheromones = json.load(f)
                print("Pheromones loaded successfully.")
    
    def save_pheromones(self):
        with open(self.pheromone_file, "w", encoding="utf-8") as f:
            json.dump(self.pheromones, f, indent=4)
            print("Pheromones saved successfully.")
    
    def train(self, tagged_folder):
        for filename in os.listdir(tagged_folder):
            if filename.endswith(".pos"):
                file_path = os.path.join(tagged_folder, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        words = line.strip("[]").split()
                        for word_pos in words:
                            if '/' in word_pos:
                                word, pos = word_pos.rsplit('/', 1)
                                self.pheromones[word][pos] += 1
        
        for word in self.pheromones:
            total = sum(self.pheromones[word].values())
            for pos in self.pheromones[word]:
                self.pheromones[word][pos] /= total
        
        self.save_pheromones()
    
    def tag_sentence(self, sentence):
        words = sentence.split()
        tagged_sentence = []
        
        for word in words:
            if word in self.pheromones:
                pos = max(self.pheromones[word], key=self.pheromones[word].get)
            else:
                pos = random.choice(list(self.pheromones.keys())) if self.pheromones else "NN"
            tagged_sentence.append(f"{word}/{pos}")
        
        return " ".join(tagged_sentence)
    
    def tag_raw_files(self, raw_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(raw_folder):
            file_path = os.path.join(raw_folder, filename)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sentences = f.readlines()
            
            tagged_sentences = [self.tag_sentence(sentence.strip()) for sentence in sentences]
            output_path = os.path.join(output_folder, filename + ".pos")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(tagged_sentences))
                print(f"Tagged file saved: {output_path}")

tagger = ACOPOSTagger()
tagged_folder = "/Users/dariadragomir/AI_siemens/POS tagging/treebank/tagged"
raw_folder = "/Users/dariadragomir/AI_siemens/POS tagging/treebank/raw"
output_folder = "/Users/dariadragomir/AI_siemens/POS tagging/treebank/tagged_predictions"

tagger.train(tagged_folder)
tagger.tag_raw_files(raw_folder, output_folder)
