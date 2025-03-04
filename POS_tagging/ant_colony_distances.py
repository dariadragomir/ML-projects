import nltk
import networkx as nx
import matplotlib.pyplot as plt
import random
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tag_sentence(sentence):
    tokens = word_tokenize(sentence)
    return pos_tag(tokens)

def build_graph(pos_tags):
    graph = nx.DiGraph()
    for i, (word, pos) in enumerate(pos_tags):
        graph.add_node(word, pos=pos)
        if i > 0: 
            graph.add_edge(pos_tags[i-1][0], word, weight=random.uniform(0.1, 1.0))
    return graph

def ant_colony_optimization(graph, iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    pheromone = {edge: 1.0 for edge in graph.edges}
    
    for _ in range(iterations):
        for edge in graph.edges:
            head, tail = edge
            weight = graph[head][tail]['weight']
            pheromone[edge] += alpha * (1.0 / weight)
        
        for edge in pheromone:
            pheromone[edge] *= (1 - evaporation_rate)
    
    return pheromone

def visualize_tree(graph, pheromone):
    pos = nx.spring_layout(graph)
    edge_labels = {edge: f"{pheromone[edge]:.2f}" for edge in graph.edges}
    
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

sentence = "Mr. Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29."
pos_tags = pos_tag_sentence(sentence)

graph = build_graph(pos_tags)
pheromone = ant_colony_optimization(graph)
visualize_tree(graph, pheromone)

