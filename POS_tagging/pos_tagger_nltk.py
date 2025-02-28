import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import treebank
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("treebank")

sentences = treebank.sents()
tagged_sentences = treebank.tagged_sents()

def pos_tag_sentence(sentence):
    tokens = word_tokenize(sentence)
    return pos_tag(tokens)

print("Sample sentence:", " ".join(sentences[0]))
print("POS-tagged sample:", tagged_sentences[0])

test_sentences = [
    "I am eating a lot of candy.",
    "Time flies like an arrow.",
    "Fruit flies like a banana.",
    "I don't like fruit flies like a banana."
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print("POS Tags:", pos_tag_sentence(sentence))
    print()

ps = PorterStemmer()
sentence = "I am eating a lot of candy."
stemmed_sentence = [ps.stem(word) for word in word_tokenize(sentence)]
print("Stemmed Sentence:", stemmed_sentence)
