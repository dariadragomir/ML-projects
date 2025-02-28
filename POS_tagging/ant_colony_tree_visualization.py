import matplotlib.pyplot as plt
import networkx as nx
import nltk
from nltk import pos_tag, word_tokenize

sentence = "Mr. Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29."

tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)

G = nx.DiGraph()

G.add_node("Sentence")

for word, pos in pos_tags:
    G.add_node(word)
    G.add_edge("Sentence", word)
    
    G.add_node(pos)
    G.add_edge(word, pos)

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42, k=0.5) 
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray", font_size=10, font_weight="bold")

plt.show()
