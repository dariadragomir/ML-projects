# create tree from input
import os
from collections import defaultdict

class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
    
    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.label) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

def parse_input(input_text):
    sentences = []
    current_sentence = []
    for token in input_text.split("\n"):
        if token.strip():
            current_sentence.append(token.strip())
        else:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def build_tree(sentences):
    root = TreeNode("Text")
    
    for i, sentence in enumerate(sentences, start=1):
        sentence_node = TreeNode(f"Sentence {i}")
        root.add_child(sentence_node)
        
        for phrase in sentence:
            words = phrase.strip("[]").split()
            for word_pos in words:
                if '/' in word_pos:
                    word, pos = word_pos.rsplit('/', 1)
                    word_node = TreeNode(f"Word: {word}")
                    pos_node = TreeNode(f"POS: {pos}")
                    word_node.add_child(pos_node)
                    sentence_node.add_child(word_node)
    
    return root

def process_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".pos"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                input_text = f.read()
            sentences = parse_input(input_text)
            tree = build_tree(sentences)
            output_path = os.path.join(output_folder, f"{filename}.tree")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(str(tree))

input_folder = "./tagged"
output_folder = "./tree"

process_files(input_folder, output_folder)
