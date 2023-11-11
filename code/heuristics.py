from collections import Counter
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd

def calculate_yules_k(text):

    tokens = word_tokenize(text)

    word_counts = Counter(tokens)
    frequency_of_frequencies = Counter(word_counts.values())

    M1 = sum(frequency_of_frequencies.values())
    M2 = sum([frequency**2 * v for frequency, v in frequency_of_frequencies.items()])
    yules_k = 10000 * (M2 - M1) / (M1**2)

    return yules_k

dialogue = pd.read_csv('../data/dialogue.csv')

#create a unique identifier column for each speaker-media combination
dialogue['speaker_media'] = dialogue['speaker'] + ' - ' + dialogue['media']

#create a list of the top 6 most frequent speakers from each media in the dialogue dataframe
top_speakers = dialogue.groupby('media')['speaker_media'].value_counts().groupby(level=0).nlargest(6).index.get_level_values(2).tolist()

yules_k_values = []

for speaker in top_speakers:

    speaker_df = dialogue[dialogue['speaker_media'] == speaker]
    speaker_df = speaker_df.dropna()

    #convert all the speaker's lines of dialogue to a single text string
    speaker_text = ' '.join(speaker_df['dialogue'].tolist())

    yules_k_value = calculate_yules_k(speaker_text)
    yules_k_values.append((speaker, yules_k_value))

print(yules_k_values)


