import re
import glob
from collections import defaultdict
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

def parse_pos_files(file_pattern):
    all_sentences = []
    word_to_tags = defaultdict(set)
    exclude_tags = [',', '.', '``', "''", '(', ')', ':', '|', '#', '$', ";", '!', '?'] 
    
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
                                    word, tag = parts
                                    if tag not in exclude_tags and '|' not in tag:
                                        current_sentence.append((word, tag))
                                        word_to_tags[word].add(tag)
                    else:
                        parts = elem.rsplit('/', 1)
                        if len(parts) == 2:
                            word, tag = parts
                            if tag not in exclude_tags and '|' not in tag:
                                current_sentence.append((word, tag))
                                word_to_tags[word].add(tag)
            
            if current_sentence:
                all_sentences.append(current_sentence)
    
    return all_sentences, word_to_tags

def compute_bigram_probabilities(sentences):
    bigram_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    for sentence in sentences:
        tags = [tag for (_, tag) in sentence]
        for a, b in zip(tags[:-1], tags[1:]):
            bigram_counts[(a, b)] += 1
            tag_counts[a] += 1
    
    probabilities = {}
    for (a, b), count in bigram_counts.items():
        if tag_counts[a] > 0:
            probabilities[(a, b)] = count / tag_counts[a]
    return probabilities

def get_transition_prob(tag1, tag2, transition_probs, default=0.1):
    return transition_probs.get((tag1, tag2), default)

def ant_colony_pos_tagging(sentence_words, pos_options, transition_probs, 
                           num_ants=10, num_iterations=50, 
                           evaporation_rate=0.1, deposit_factor=1.0):
    
    best_sequence = None
    best_fitness = float('-inf')
    
    all_tags = set()
    for tags in pos_options.values():
        all_tags.update(tags)
    
    pheromones = {}
    for tag1 in all_tags:
        for tag2 in all_tags:
            pheromones[(tag1, tag2)] = 1.0

    for iteration in range(num_iterations):
        for ant in range(num_ants):
            current_sequence = []
            fitness = 0.0
            word = sentence_words[0]
            options = pos_options.get(word, list(all_tags))
            chosen_tag = random.choice(options)
            current_sequence.append(chosen_tag)
            
            for i in range(1, len(sentence_words)):
                word = sentence_words[i]
                options = pos_options.get(word, list(all_tags))
                prev_tag = current_sequence[-1]
                weights = []
                for tag in options:
                    pheromone = pheromones.get((prev_tag, tag), 1.0)
                    trans_prob = get_transition_prob(prev_tag, tag, transition_probs)
                    weights.append(pheromone * trans_prob)
                chosen_tag = random.choices(options, weights=weights, k=1)[0]
                current_sequence.append(chosen_tag)
                fitness += get_transition_prob(prev_tag, chosen_tag, transition_probs)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = current_sequence.copy()
            
            for i in range(1, len(current_sequence)):
                prev_tag = current_sequence[i-1]
                curr_tag = current_sequence[i]
                pheromones[(prev_tag, curr_tag)] += deposit_factor * fitness

        for key in pheromones:
            pheromones[key] *= (1 - evaporation_rate)
    
    return best_sequence

def parse_text(text):
    sentences = sent_tokenize(text)
    return [word_tokenize(sent) for sent in sentences]

if __name__ == "__main__":
    file_pattern = 'treebank/tagged/*.pos'
    sentences, word_to_tags = parse_pos_files(file_pattern)
    bigram_probs = compute_bigram_probabilities(sentences)
    
    print("Computed Bigram Probabilities:")
    for (a, b), prob in sorted(bigram_probs.items(), key=lambda x: (-x[1], x[0])):
        print(f"P({b}|{a}) = {prob:.4f}")
    
    all_tags = set()
    for tags in word_to_tags.values():
        all_tags.update(tags)
    pos_options = {word: list(tags) for word, tags in word_to_tags.items()}
    
    input_text = """
    Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.
    Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group!
    What's the state-of-the-art in 2023?
    """
    parsed_sentences = parse_text(input_text)
    
    for i, sentence_words in enumerate(parsed_sentences, 1):
        best_tag_sequence = ant_colony_pos_tagging(sentence_words, pos_options, bigram_probs,
                                                    num_ants=20, num_iterations=100,
                                                    evaporation_rate=0.1, deposit_factor=1.0)
        print(f"\nSentence {i}: {' '.join(sentence_words)}")
        print("POS Tag Sequence:", best_tag_sequence)

# output: 1. probabilities for bigrams
# 2. predicted POS tags 

'''
Computed Bigram Probabilities:
P(NNP|SYM) = 1.0000
P(DT|PDT) = 0.8889
P(VB|MD) = 0.8198
P(JJ|RBS) = 0.7143
P(VBZ|EX) = 0.5909
P(VB|TO) = 0.5803
P(NN|WP$) = 0.5714
P(VBZ|FW) = 0.5000
P(IN|UH) = 0.5000
P(PRP|UH) = 0.5000
P(NN|DT) = 0.4750
P(NN|JJ) = 0.4581
P(NNP|NNP) = 0.4518
P(NN|PRP$) = 0.4399
P(NNP|NNPS) = 0.4260
P(NN|POS) = 0.4191
P(JJ|RBR) = 0.3824
P(IN|VBN) = 0.3815
P(NNS|WP$) = 0.3571
P(IN|JJR) = 0.3466
P(DT|WRB) = 0.3202
P(DT|IN) = 0.3196
P(IN|NNS) = 0.2981
P(VBZ|WDT) = 0.2966
P(IN|NN) = 0.2910
P(NN|JJS) = 0.2865
P(VBD|PRP) = 0.2710
P(IN|RP) = 0.2617
P(VBZ|WP) = 0.2573
P(DT|RP) = 0.2523
P(DT|FW) = 0.2500
P(NN|FW) = 0.2500
P(DT|LS) = 0.2500
P(JJ|LS) = 0.2500
P(RB|LS) = 0.2500
P(IN|RBR) = 0.2500
P(NNS|JJ) = 0.2486
P(DT|VB) = 0.2452
P(VBD|WP) = 0.2365
P(NN|CD) = 0.2357
P(JJ|PRP$) = 0.2298
P(VBZ|PRP) = 0.2218
P(CD|CD) = 0.2182
P(VBP|WP) = 0.2116
P(NN|JJR) = 0.2090
P(VBD|WDT) = 0.2090
P(NNS|PRP$) = 0.2063
P(JJ|DT) = 0.2061
P(JJ|POS) = 0.2059
P(NNS|JJR) = 0.2011
P(RB|RBS) = 0.2000
P(DT|VBG) = 0.1940
P(VBP|EX) = 0.1932
P(DT|VBD) = 0.1873
P(VBP|PRP) = 0.1827
P(NNS|JJS) = 0.1798
P(DT|VBZ) = 0.1751
P(VBN|VBP) = 0.1737
P(VBP|WDT) = 0.1708
P(IN|JJS) = 0.1629
P(NNS|CD) = 0.1624
P(RB|MD) = 0.1618
P(VBN|VBZ) = 0.1582
P(PRP|WRB) = 0.1573
P(NNP|CC) = 0.1570
P(NN|NN) = 0.1493
P(NNP|IN) = 0.1492
P(NN|VBG) = 0.1480
P(RB|VBP) = 0.1472
P(IN|VBG) = 0.1419
P(JJ|JJS) = 0.1404
P(IN|RB) = 0.1396
P(IN|VBD) = 0.1392
P(DT|VBP) = 0.1388
P(RB|VBZ) = 0.1375
P(VBD|EX) = 0.1364
P(NNS|POS) = 0.1336
P(MD|PRP) = 0.1316
P(DT|TO) = 0.1312
P(MD|WDT) = 0.1303
P(NNP|POS) = 0.1299
P(IN|VB) = 0.1299
P(NNP|DT) = 0.1293
P(PRP|LS) = 0.1250
P(VBN|LS) = 0.1250
P(NN|CC) = 0.1221
P(DT|CC) = 0.1142
P(IN|VBP) = 0.1138
P(CD|TO) = 0.1119
P(PRP$|PDT) = 0.1111
P(NN|IN) = 0.1101
P(VBP|NNS) = 0.1069
P(TO|VBN) = 0.1068
P(IN|CD) = 0.1057
P(JJ|CC) = 0.1054
P(JJ|RB) = 0.1031
P(VBN|RB) = 0.1031
P(RB|RBR) = 0.1029
P(IN|VBZ) = 0.1017
P(VB|RB) = 0.1013
P(JJ|WRB) = 0.1011
P(VBN|VBD) = 0.1011
P(IN|NNPS) = 0.0987
P(JJ|IN) = 0.0977
P(NNS|VBG) = 0.0966
P(VBD|NNS) = 0.0963
P(NNS|NN) = 0.0961
P(VBN|VB) = 0.0942
P(TO|VBG) = 0.0918
P(CD|IN) = 0.0903
P(JJ|VB) = 0.0894
P(CC|NNS) = 0.0844
P(JJ|VBP) = 0.0842
P(RB|VBD) = 0.0839
P(DT|VBN) = 0.0837
P(VBG|VBP) = 0.0812
P(DT|RB) = 0.0811
P(NN|VBN) = 0.0809
P(CC|NNPS) = 0.0807
P(RB|RB) = 0.0804
P(VBD|NNP) = 0.0795
P(NNS|DT) = 0.0777
P(JJ|JJ) = 0.0754
P(NNP|VBD) = 0.0754
P(PRP|VBD) = 0.0750
P(NNS|RP) = 0.0748
P(PRP$|RP) = 0.0748
P(JJ|VBG) = 0.0733
P(NNP|WRB) = 0.0730
P(JJ|VBZ) = 0.0730
P(NN|NNP) = 0.0719
P(JJ|WP$) = 0.0714
P(NNS|CC) = 0.0710
P(VBD|RB) = 0.0702
P(NNP|VBZ) = 0.0697
P(NN|VB) = 0.0679
P(NNS|WRB) = 0.0674
P(CD|VBD) = 0.0668
P(JJ|JJR) = 0.0661
P(IN|JJ) = 0.0656
P(TO|RP) = 0.0654
P(VBD|NN) = 0.0650
P(NNS|IN) = 0.0630
P(VBD|NNPS) = 0.0628
P(JJ|VBN) = 0.0621
P(CD|JJS) = 0.0618
P(NN|WRB) = 0.0618
P(TO|VBP) = 0.0615
P(CD|CC) = 0.0613
P(TO|VBD) = 0.0609
P(RB|WDT) = 0.0607
P(VBZ|NN) = 0.0593
P(DT|RBR) = 0.0588
P(NNP|PRP$) = 0.0587
P(IN|CC) = 0.0578
P(VBG|VBZ) = 0.0574
P(CC|NN) = 0.0571
P(IN|RBS) = 0.0571
P(MD|EX) = 0.0568
P(IN|NNP) = 0.0565
P(JJ|RP) = 0.0561
P(TO|VBZ) = 0.0546
P(MD|WP) = 0.0539
P(PRP|WP) = 0.0539
P(POS|NNP) = 0.0528
P(RB|PRP) = 0.0523
P(RB|CC) = 0.0520
P(JJ|VBD) = 0.0513
P(RB|VBN) = 0.0508
P(NNS|VB) = 0.0475
P(CC|NNP) = 0.0474
P(RB|RP) = 0.0467
P(TO|NN) = 0.0461
P(TO|NNS) = 0.0457
P(RB|WP) = 0.0456
P(JJ|CD) = 0.0454
P(POS|NNPS) = 0.0448
P(VBP|NNPS) = 0.0448
P(VBZ|NNP) = 0.0443
P(PRP|CC) = 0.0441
P(RB|VB) = 0.0439
P(CC|CD) = 0.0438
P(NN|VBZ) = 0.0433
P(RB|NNS) = 0.0433
P(PRP|VBP) = 0.0432
P(CD|RB) = 0.0429
P(VBZ|RB) = 0.0429
P(CC|JJR) = 0.0423
P(PRP|VB) = 0.0416
P(DT|WP) = 0.0415
P(TO|CD) = 0.0414
P(NNP|TO) = 0.0408
P(NNP|VB) = 0.0408
P(DT|CD) = 0.0408
P(NN|VBD) = 0.0405
P(NNS|VBN) = 0.0400
P(RB|VBG) = 0.0398
P(VBD|CC) = 0.0397
P(IN|PRP) = 0.0397
P(PRP$|VB) = 0.0396
P(PRP|WDT) = 0.0382
P(NNP|VBG) = 0.0377
P(NNP|JJ) = 0.0376
P(VBG|RB) = 0.0376
P(NN|RP) = 0.0374
P(DT|JJR) = 0.0370
P(TO|VB) = 0.0369
P(VBN|RBR) = 0.0368
P(NNP|VBN) = 0.0367
P(MD|NNPS) = 0.0359
P(PRP|VBG) = 0.0350
P(PRP$|IN) = 0.0343
P(RB|JJS) = 0.0337
P(VBN|WRB) = 0.0337
P(CD|NNP) = 0.0334
P(CD|VB) = 0.0333
P(VBP|RB) = 0.0333
P(PRP|IN) = 0.0333
P(VB|CC) = 0.0331
P(DT|NNS) = 0.0331
P(MD|NNS) = 0.0329
P(DT|NNP) = 0.0324
P(PRP|VBZ) = 0.0320
P(NN|VBP) = 0.0319
P(CD|DT) = 0.0316
P(DT|NNPS) = 0.0314
P(NN|NNPS) = 0.0314
P(VBG|IN) = 0.0311
P(NN|NNS) = 0.0310
P(NNP|NN) = 0.0307
P(CD|POS) = 0.0306
P(JJ|NNS) = 0.0306
P(DT|NN) = 0.0305
P(CD|PRP$) = 0.0300
P(NNP|CD) = 0.0294
P(NNP|RBR) = 0.0294
P(JJ|TO) = 0.0289
P(NNS|TO) = 0.0289
P(CD|VBG) = 0.0288
P(VBN|VBN) = 0.0287
P(NNS|VBD) = 0.0286
P(VBN|RBS) = 0.0286
P(NN|TO) = 0.0284
P(DT|JJS) = 0.0281
P(RB|WRB) = 0.0281
P(CC|RP) = 0.0280
P(CD|RP) = 0.0280
P(VBZ|RP) = 0.0280
P(NNP|VBP) = 0.0273
P(VBG|NNS) = 0.0270
P(VBZ|NNPS) = 0.0269
P(PRP$|VBG) = 0.0267
P(NNS|VBP) = 0.0266
P(VBN|NNS) = 0.0257
P(NNS|NNP) = 0.0256
P(TO|RB) = 0.0255
P(RB|NN) = 0.0254
P(CD|VBN) = 0.0254
P(TO|JJ) = 0.0252
P(VBG|CC) = 0.0247
P(VBG|VBD) = 0.0244
P(CD|JJ) = 0.0242
P(CD|VBZ) = 0.0240
P(VBZ|CC) = 0.0238
P(NNS|NNS) = 0.0238
P(WDT|NNS) = 0.0238
P(PRP$|VBD) = 0.0237
P(POS|NN) = 0.0236
P(CC|VBN) = 0.0230
P(RB|EX) = 0.0227
P(VBG|JJS) = 0.0225
P(DT|WDT) = 0.0225
P(IN|WRB) = 0.0225
P(VBP|WRB) = 0.0225
P(VBZ|WRB) = 0.0225
P(NNS|NNPS) = 0.0224
P(RB|NNPS) = 0.0224
P(NNP|RB) = 0.0223
P(IN|IN) = 0.0222
P(CC|RBR) = 0.0221
P(VB|RBR) = 0.0221
P(CC|VBG) = 0.0219
P(CC|JJ) = 0.0214
P(VBN|VBG) = 0.0212
P(NNP|JJR) = 0.0212
P(TO|JJR) = 0.0212
P(JJS|POS) = 0.0208
P(NNP|WP) = 0.0207
P(TO|PRP) = 0.0204
P(NN|RB) = 0.0198
P(PRP|RB) = 0.0198
P(MD|NN) = 0.0193
P(VBG|VBN) = 0.0188
P(PRP|RP) = 0.0187
P(NNPS|NNP) = 0.0181
P(JJR|RB) = 0.0174
P(VB|JJS) = 0.0169
P(VBN|JJS) = 0.0169
P(CD|WRB) = 0.0169
P(PRP$|WRB) = 0.0169
P(VBZ|NNS) = 0.0168
P(VBN|CC) = 0.0168
P(IN|WP) = 0.0166
P(VBG|VB) = 0.0165
P(PRP|VBN) = 0.0165
P(VBG|NN) = 0.0158
P(NNS|WDT) = 0.0157
P(WDT|NN) = 0.0157
P(JJ|NNP) = 0.0157
P(PRP$|TO) = 0.0151
P(JJR|VBG) = 0.0151
P(NNS|VBZ) = 0.0151
P(CC|PRP) = 0.0150
P(PRP$|CC) = 0.0150
P(RB|NNP) = 0.0149
P(PRP|NN) = 0.0148
P(PRP|RBR) = 0.0147
P(TO|RBR) = 0.0147
P(VBG|RBR) = 0.0147
P(JJ|NN) = 0.0146
P(RB|CD) = 0.0141
P(RB|IN) = 0.0138
P(PRP|NNS) = 0.0138
P(PRP$|VBN) = 0.0136
P(CC|RB) = 0.0135
P(VBN|NNPS) = 0.0135
P(NNP|NNS) = 0.0134
P(PRP|JJR) = 0.0132
P(NNS|RB) = 0.0131
P(JJS|PRP$) = 0.0131
P(VBN|NN) = 0.0129
P(CC|VB) = 0.0126
P(CD|NNS) = 0.0125
P(NN|WP) = 0.0124
P(PRP$|WP) = 0.0124
P(VBP|CC) = 0.0123
P(VBN|POS) = 0.0123
P(DT|PRP) = 0.0120
P(MD|NNP) = 0.0120
P(PRP$|VBZ) = 0.0118
P(MD|CC) = 0.0115
P(WP|NNS) = 0.0113
P(NNP|JJS) = 0.0112
P(JJ|WDT) = 0.0112
P(NNP|WDT) = 0.0112
P(TO|WRB) = 0.0112
P(VBD|WRB) = 0.0112
P(VBG|DT) = 0.0110
P(MD|RB) = 0.0110
P(RP|VB) = 0.0110
P(POS|NNS) = 0.0110
P(VBN|DT) = 0.0108
P(CD|VBP) = 0.0106
P(PRP$|VBP) = 0.0106
P(JJR|CC) = 0.0106
P(VBD|CD) = 0.0104
P(DT|JJ) = 0.0104
P(CD|NN) = 0.0103
P(VB|PRP) = 0.0102
P(VB|VB) = 0.0102
P(IN|DT) = 0.0101
P(JJS|DT) = 0.0097
P(VBN|CD) = 0.0095
P(RB|DT) = 0.0094
P(CC|VBZ) = 0.0094
P(RP|VBN) = 0.0094
P(EX|RP) = 0.0093
P(NNP|RP) = 0.0093
P(VBG|RP) = 0.0093
P(IN|WDT) = 0.0090
P(NNPS|NNPS) = 0.0090
P(TO|NNPS) = 0.0090
P(VBG|NNPS) = 0.0090
P(WDT|NNPS) = 0.0090
P(WP|NNPS) = 0.0090
P(VBG|POS) = 0.0086
P(CC|VBP) = 0.0083
P(JJR|VBP) = 0.0083
P(NNS|WP) = 0.0083
P(RB|TO) = 0.0083
P(JJR|VB) = 0.0082
P(JJR|VBD) = 0.0082
P(RP|VBG) = 0.0082
P(TO|NNP) = 0.0082
P(JJR|VBZ) = 0.0080
P(RB|JJR) = 0.0079
P(RBR|VB) = 0.0078
P(VBN|PRP$) = 0.0078
P(RBR|RB) = 0.0078
P(VBZ|CD) = 0.0077
P(WDT|NNP) = 0.0075
P(VBD|POS) = 0.0074
P(MD|RBR) = 0.0074
P(NN|RBR) = 0.0074
P(NNS|RBR) = 0.0074
P(VBD|RBR) = 0.0074
P(VBP|RBR) = 0.0074
P(VBZ|RBR) = 0.0074
P(WRB|RBR) = 0.0074
P(JJ|PRP) = 0.0072
P(PRP|PRP) = 0.0072
P(TO|CC) = 0.0071
P(CC|VBD) = 0.0069
P(RP|VBP) = 0.0068
P(MD|CD) = 0.0067
P(NN|WDT) = 0.0067
P(VBG|CD) = 0.0064
P(VBG|TO) = 0.0064
P(JJR|IN) = 0.0063
P(NNPS|POS) = 0.0061
P(RB|POS) = 0.0061
P(PRP|JJ) = 0.0060
P(RB|JJ) = 0.0060
P(VBZ|DT) = 0.0060
P(VBP|NN) = 0.0060
P(IN|TO) = 0.0060
P(VBP|NNP) = 0.0059
P(JJR|DT) = 0.0059
P(VBG|JJ) = 0.0059
P(VBN|IN) = 0.0058
P(VB|NNS) = 0.0057
P(CC|JJS) = 0.0056
P(JJR|JJS) = 0.0056
P(NNPS|JJS) = 0.0056
P(PRP|JJS) = 0.0056
P(VBD|JJS) = 0.0056
P(VBP|JJS) = 0.0056
P(VBZ|JJS) = 0.0056
P(EX|WRB) = 0.0056
P(MD|WRB) = 0.0056
P(PDT|WRB) = 0.0056
P(RBS|WRB) = 0.0056
P(RP|WRB) = 0.0056
P(VBG|WRB) = 0.0056
P(RP|VBD) = 0.0056
P(PRP|CD) = 0.0055
P(VBG|VBG) = 0.0055
P(DT|MD) = 0.0054
P(VBN|NNP) = 0.0054
P(EX|CC) = 0.0053
P(CD|JJR) = 0.0053
P(VB|JJR) = 0.0053
P(VBD|JJR) = 0.0053
P(VBG|PRP$) = 0.0052
P(PRP|TO) = 0.0050
P(CC|POS) = 0.0049
P(JJR|TO) = 0.0046
P(WP|NNP) = 0.0046
P(RBR|VBP) = 0.0046
P(VBD|VBP) = 0.0046
P(CD|WDT) = 0.0045
P(VBN|WDT) = 0.0045
P(CD|NNPS) = 0.0045
P(JJ|NNPS) = 0.0045
P(PRP$|NNPS) = 0.0045
P(VBG|NNP) = 0.0045
P(WRB|CC) = 0.0044
P(JJS|IN) = 0.0043
P(VBP|CD) = 0.0043
P(VBD|VBD) = 0.0043
P(RP|VBZ) = 0.0042
P(WRB|VBZ) = 0.0042
P(VBD|VBN) = 0.0042
P(JJR|PRP) = 0.0042
P(NN|PRP) = 0.0042
P(NNP|PRP) = 0.0042
P(VBG|PRP) = 0.0042
P(CD|WP) = 0.0041
P(JJ|WP) = 0.0041
P(PDT|WP) = 0.0041
P(POS|WP) = 0.0041
P(RBS|WP) = 0.0041
P(TO|WP) = 0.0041
P(VBN|WP) = 0.0041
P(WP|NN) = 0.0041
P(WRB|VB) = 0.0039
P(VBZ|VBP) = 0.0038
P(JJR|VBN) = 0.0038
P(VBN|JJ) = 0.0036
P(WDT|IN) = 0.0036
P(VBN|PRP) = 0.0036
P(WP|VB) = 0.0035
P(NNPS|DT) = 0.0034
P(RBR|VBG) = 0.0034
P(JJR|NNS) = 0.0034
P(WRB|NNS) = 0.0034
P(PRP|NNP) = 0.0034
P(TO|IN) = 0.0033
P(RBR|VBZ) = 0.0033
P(VB|VBZ) = 0.0033
P(VBD|VBZ) = 0.0033
P(VBZ|VBZ) = 0.0033
P(RBR|VBD) = 0.0033
P(VB|VBD) = 0.0033
P(TO|MD) = 0.0032
P(WRB|NN) = 0.0032
P(WRB|RB) = 0.0032
P(VBZ|VB) = 0.0031
P(WDT|CD) = 0.0031
P(WRB|CD) = 0.0031
P(MD|VBP) = 0.0030
P(WP|VBZ) = 0.0028
P(PDT|VB) = 0.0027
P(VBD|VB) = 0.0027
P(RBR|CC) = 0.0026
P(EX|JJR) = 0.0026
P(JJR|JJR) = 0.0026
P(LS|JJR) = 0.0026
P(MD|JJR) = 0.0026
P(VBG|JJR) = 0.0026
P(VBN|JJR) = 0.0026
P(WRB|JJR) = 0.0026
P(JJR|PRP$) = 0.0026
P(EX|RB) = 0.0025
P(PRP$|RB) = 0.0025
P(DT|POS) = 0.0025
P(IN|POS) = 0.0025
P(JJR|POS) = 0.0025
P(RBS|POS) = 0.0025
P(VBZ|POS) = 0.0025
P(JJR|NN) = 0.0024
P(VB|NN) = 0.0024
P(VBZ|JJ) = 0.0024
P(RP|PRP) = 0.0024
P(RBR|VBN) = 0.0024
P(WP|VBN) = 0.0024
P(RBS|DT) = 0.0023
P(WRB|VBD) = 0.0023
P(VBP|VBP) = 0.0023
P(WRB|VBP) = 0.0023
P(EX|WDT) = 0.0022
P(JJS|WDT) = 0.0022
P(RBR|WDT) = 0.0022
P(TO|WDT) = 0.0022
P(VBD|JJ) = 0.0022
P(JJS|CC) = 0.0022
P(NNPS|CC) = 0.0022
P(IN|MD) = 0.0022
P(WDT|RB) = 0.0021
P(WP|RB) = 0.0021
P(WP|IN) = 0.0021
P(VBD|VBG) = 0.0021
P(VBZ|VBG) = 0.0021
P(WP|VBG) = 0.0021
P(NNPS|IN) = 0.0020
P(EX|VBD) = 0.0020
P(VBP|VBN) = 0.0019
P(WRB|VBN) = 0.0019
P(VBD|DT) = 0.0018
P(NNS|PRP) = 0.0018
P(WP|PRP) = 0.0018
P(WP|CC) = 0.0018
P(RBR|DT) = 0.0017
P(DT|DT) = 0.0016
P(EX|VB) = 0.0016
P(JJS|VB) = 0.0016
P(WRB|NNP) = 0.0016
P(JJR|CD) = 0.0015
P(VB|VBP) = 0.0015
P(WP|VBP) = 0.0015
P(RBR|NNS) = 0.0015
P(MD|DT) = 0.0015
P(VB|NNP) = 0.0015
P(EX|VBN) = 0.0014
P(VBZ|VBN) = 0.0014
P(WRB|IN) = 0.0014
P(VBP|JJ) = 0.0014
P(RBR|TO) = 0.0014
P(VBN|TO) = 0.0014
P(MD|VBG) = 0.0014
P(PRP|DT) = 0.0013
P(PRP$|NNS) = 0.0013
P(MD|VBD) = 0.0013
P(VBZ|VBD) = 0.0013
P(IN|PRP$) = 0.0013
P(NNPS|PRP$) = 0.0013
P(RB|PRP$) = 0.0013
P(RBS|PRP$) = 0.0013
P(VBZ|PRP$) = 0.0013
P(JJS|CD) = 0.0012
P(WP|CD) = 0.0012
P(PRP|POS) = 0.0012
P(WP|POS) = 0.0012
P(JJR|JJ) = 0.0012
P(WRB|JJ) = 0.0012
P(CC|IN) = 0.0012
P(CD|PRP) = 0.0012
P(EX|IN) = 0.0011
P(VBD|IN) = 0.0011
P(VBP|DT) = 0.0011
P(JJ|MD) = 0.0011
P(JJS|MD) = 0.0011
P(NNP|MD) = 0.0011
P(NNS|MD) = 0.0011
P(PRP|MD) = 0.0011
P(RBR|MD) = 0.0011
P(VBG|MD) = 0.0011
P(PRP$|NN) = 0.0010
P(WP|VBD) = 0.0010
P(WP$|NNS) = 0.0009
P(EX|VBZ) = 0.0009
P(JJS|VBZ) = 0.0009
P(MD|VBZ) = 0.0009
P(WDT|VBN) = 0.0009
P(EX|NN) = 0.0009
P(POS|CD) = 0.0009
P(CC|TO) = 0.0009
P(PDT|IN) = 0.0009
P(RBR|IN) = 0.0009
P(WDT|CC) = 0.0009
P(NNPS|JJ) = 0.0009
P(WP|DT) = 0.0009
P(RBR|NN) = 0.0008
P(MD|VB) = 0.0008
P(NNPS|VB) = 0.0008
P(RBS|VB) = 0.0008
P(VBP|VB) = 0.0008
P(WDT|VB) = 0.0008
P(EX|VBP) = 0.0008
P(PDT|VBP) = 0.0008
P(WDT|VBP) = 0.0008
P(EX|NNS) = 0.0008
P(NNPS|RB) = 0.0007
P(MD|JJ) = 0.0007
P(RBR|JJ) = 0.0007
P(NNPS|VBG) = 0.0007
P(PDT|VBG) = 0.0007
P(RBS|VBG) = 0.0007
P(VB|VBG) = 0.0007
P(WRB|VBG) = 0.0007
P(PDT|VBD) = 0.0007
P(VBP|VBD) = 0.0007
P(RBR|CD) = 0.0006
P(CC|DT) = 0.0006
P(RBR|PRP) = 0.0006
P(WDT|PRP) = 0.0006
P(LS|NNS) = 0.0006
P(JJS|NN) = 0.0005
P(NNPS|VBZ) = 0.0005
P(PDT|VBZ) = 0.0005
P(UH|VBZ) = 0.0005
P(VBP|VBZ) = 0.0005
P(LS|VBN) = 0.0005
P(MD|VBN) = 0.0005
P(RBS|VBN) = 0.0005
P(VB|VBN) = 0.0005
P(WRB|TO) = 0.0005
P(JJR|NNP) = 0.0004
P(WP$|NNP) = 0.0004
P(CC|CC) = 0.0004
P(LS|CC) = 0.0004
P(RBS|CC) = 0.0004
P(VB|IN) = 0.0004
P(VBZ|IN) = 0.0004
P(WP$|VB) = 0.0004
P(RBS|NNS) = 0.0004
P(TO|DT) = 0.0004
P(WDT|DT) = 0.0004
P(JJS|RB) = 0.0004
P(PDT|RB) = 0.0004
P(RBS|RB) = 0.0004
P(JJS|JJ) = 0.0003
P(WP|JJ) = 0.0003
P(RP|NN) = 0.0003
P(PRP$|NNP) = 0.0003
P(JJS|VBD) = 0.0003
P(RBS|VBD) = 0.0003
P(UH|VBD) = 0.0003
P(PRP$|CD) = 0.0003
P(VB|CD) = 0.0003
P(WP$|CD) = 0.0003
P(VBP|IN) = 0.0002
P(JJS|NNS) = 0.0002
P(NNPS|NNS) = 0.0002
P(RP|NNS) = 0.0002
P(UH|NNS) = 0.0002
P(PDT|JJ) = 0.0002
P(PRP$|JJ) = 0.0002
P(RP|JJ) = 0.0002
P(WDT|JJ) = 0.0002
P(FW|NN) = 0.0002
P(LS|NN) = 0.0002
P(WP$|NN) = 0.0002
P(EX|DT) = 0.0001
P(FW|DT) = 0.0001
P(PRP$|DT) = 0.0001
P(RP|NNP) = 0.0001
P(SYM|NNP) = 0.0001
P(FW|IN) = 0.0001
P(MD|IN) = 0.0001
P(WP$|IN) = 0.0001
P(NNPS|NN) = 0.0001
P(RBS|NN) = 0.0001

Sentence 1: Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .
POS Tag Sequence: ['NNP', 'NNP', 'NNP', 'CD', 'NNS', 'JJ', 'WP$', 'NN', 'VB', 'DT', 'NN', 'IN', 'DT', 'JJ', 'NN', 'NN', 'CD', 'NNS']

Sentence 2: Mr. Vinken is chairman of Elsevier N.V. , the Dutch publishing group !
POS Tag Sequence: ['NNP', 'NNP', 'VBZ', 'NN', 'IN', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NN', 'NN', 'IN']

Sentence 3: What 's the state-of-the-art in 2023 ?
POS Tag Sequence: ['WP', 'VBZ', 'DT', 'NN', 'IN', 'RBS', 'JJ']
'''
