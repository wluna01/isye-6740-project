import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from collections import Counter

def get_sentence_lengths(dialogues):

    # Tokenizing each dialogue into sentences
    #sentences = [sentence for dialogue in dialogues if isinstance(dialogue, str) for sentence in sent_tokenize(str(dialogue_list))]
    # Calculating sentence lengths in words
    #sentence_lengths = [len(sentence.split()) for sentence in sentences]
    # Creating a distribution
    #length_distribution = Counter(sentence_lengths)

    #return length_distribution

    # Flattened list to store all sentence lengths
    all_sentence_lengths = []

    # Tokenizing each dialogue into sentences
    for dialogue in dialogues:
        if isinstance(dialogue, str):
            sentences = sent_tokenize(dialogue)
            # Extending the list with the length of each sentence
            all_sentence_lengths.extend(len(sentence.split()) for sentence in sentences)

    return all_sentence_lengths
