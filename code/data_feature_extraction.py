import pandas as pd
from nltk.tokenize import sent_tokenize


#read in diaologue data as a pandas df
dialogue = pd.read_csv('../data/dialogue.csv')
dialogue = dialogue.dropna(subset=['dialogue'])
#create a new column that contains a list of sentences from the dialogue column
dialogue['sentences'] = dialogue['dialogue'].apply(lambda text: sent_tokenize(text))
#create a new column that contains the length of the list of sentences in the sentences column
dialogue['num_sentences'] = dialogue['sentences'].apply(lambda text: len(text))
#create a new column that contains the number of words in the dialogue column
dialogue['num_words'] = dialogue['dialogue'].apply(lambda text: len(text.split()))
#create a new column that contains the average number of words per sentence in the dialogue column
dialogue['avg_words_per_sentence'] = dialogue['num_words']/dialogue['num_sentences']

dialogue.to_csv('../data/dialogue.csv', index=False)

#print(dialogue.sample(5))
