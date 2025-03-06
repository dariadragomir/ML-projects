# sense in the leaves, more cycles
import numpy as np
import random
from nltk.corpus import wordnet, stopwords

class ant_colony:
    def __init__(self, text, num_ants=10, cycles=100, evaporation=0.1, pheromone_deposit=1.0):
        self.text = text.split()
        self.graph = self.build_graph()
        self.num_ants = num_ants
        self.cycles = cycles
        self.evaporation = evaporation
        self.pheromone_deposit = pheromone_deposit
        
    def build_graph(self):
        graph = {}
        for word in self.text:
            senses = wordnet.synsets(word)
            graph[word] = {sense: {'energy': random.randint(5, 60), 'pheromone': {}} for sense in senses}
            for s1 in senses:
                for s2 in senses:
                    if s1 != s2:
                        similarity = self.lesk_similarity(s1, s2)
                        graph[word][s1]['pheromone'][s2] = similarity 
        return graph
    
    def lesk_similarity(self, sense1, sense2):
        def_words1 = set(sense1.definition().split()) - set(stopwords.words('english'))
        def_words2 = set(sense2.definition().split()) - set(stopwords.words('english'))
        return len(def_words1 & def_words2)

    def run(self):
        for _ in range(self.cycles):
            for _ in range(self.num_ants):
                self.simulate_ant()
            self.evaporate_pheromone()
        return self.select_best_senses()
    
    def simulate_ant(self):
        for word in self.graph:
            senses = list(self.graph[word].keys())
            if not senses:
                continue  
            probabilities = [self.compute_move_probability(word, s) for s in senses]
            if sum(probabilities) == 0:  # avoid zero weight issue
                probabilities = [1.0 / len(senses)] * len(senses) 
            chosen_sense = random.choices(senses, probabilities)[0]
            self.deposit_pheromone(word, chosen_sense)

    
    def compute_move_probability(self, word, sense):
        pheromones = self.graph[word][sense]['pheromone'].values()
        return sum(pheromones) / (1 + sum(pheromones)) if pheromones else 1.0
    
    def deposit_pheromone(self, word, sense):
        for other_sense in self.graph[word][sense]['pheromone']:
            self.graph[word][sense]['pheromone'][other_sense] += self.pheromone_deposit
    
    def evaporate_pheromone(self):
        for word in self.graph:
            for sense in self.graph[word]:
                for other_sense in self.graph[word][sense]['pheromone']:
                    self.graph[word][sense]['pheromone'][other_sense] *= (1 - self.evaporation)
    
    def select_best_senses(self):
        best_senses = {}
        for word in self.graph:
            senses = list(self.graph[word].keys())
            if senses:
                best_sense = max(senses, key=lambda s: sum(self.graph[word][s]['pheromone'].values()))
                best_senses[word] = best_sense
        return best_senses

text = "The bank can guarantee deposits will be safe"
aco = ant_colony(text)
best_senses = aco.run()
print(best_senses)
