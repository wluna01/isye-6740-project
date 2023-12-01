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
#round it to two decimal places
dialogue['avg_words_per_sentence'] = dialogue['avg_words_per_sentence'].round(2)
#create a new column the contains the number of characters in the dialogue column
dialogue['num_chars'] = dialogue['dialogue'].apply(lambda text: len(text))
#create a new column that contains the average number of characters per word in the dialogue column
dialogue['avg_chars_per_word'] = dialogue['num_chars']/dialogue['num_words']
#round it to two decimal places
dialogue['avg_chars_per_word'] = dialogue['avg_chars_per_word'].round(2)
#find the six most commonly occurring speakers for each media
top_six = dialogue.groupby('media')['speaker'].value_counts().groupby(level=0).nlargest(6).reset_index(level=0, drop=True).reset_index()
#print(top_six)
#add true false column for whether or not the speaker is in the top six for their media
dialogue['is_top_six'] = dialogue.apply(lambda x: x['speaker'] in top_six[top_six['media'] == x['media']]['speaker'].tolist(), axis=1)

dialogue.to_csv('../data/dialogue.csv', index=False)

#print(dialogue.sample(5))
