from collections import Counter
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import pandas as pd
from collections import defaultdict

def calculate_heuristics(text):

    tokens = word_tokenize(text)    
    #yule's k
    word_counts = Counter(tokens)
    frequency_of_frequencies = Counter(word_counts.values())

    M1 = sum(frequency_of_frequencies.values())
    M2 = sum([frequency**2 * v for frequency, v in frequency_of_frequencies.items()])
    yules_k = 10000 * (M2 - M1) / (M1**2)

    #simpson's d
    freq_dist = FreqDist(tokens)
    N = len(tokens)
    simpsons_d = 1 - sum((n/N)**2 for n in freq_dist.values())

    return yules_k, simpsons_d

dialogue = pd.read_csv('../data/dialogue.csv')

#create a unique identifier column for each speaker-media combination
dialogue['speaker_media'] = dialogue['speaker'] + ' - ' + dialogue['media']

#create a list of the top 6 most frequent speakers from each media in the dialogue dataframe
top_speakers = dialogue.groupby('media')['speaker_media'].value_counts().groupby(level=0).nlargest(6).index.get_level_values(2).tolist()

heuristics = []

for speaker in top_speakers:
    print(speaker)
    speaker_df = dialogue[dialogue['speaker_media'] == speaker]
    speaker_df = speaker_df.dropna()

    #convert all the speaker's lines of dialogue to a single text string
    speaker_text = ' '.join(speaker_df['dialogue'].tolist())

    yules_k, simpsons_d = calculate_heuristics(speaker_text)

    speaker_name = speaker.split(' - ')[0]
    media = speaker.split(' - ')[1]

    heuristics.append((speaker_name, media, yules_k, simpsons_d))

#save the heuristics to a csv file
heuristics_df = pd.DataFrame(heuristics, columns=['speaker', 'media', 'yules_k', 'simpsons_d'])
heuristics_df.to_csv('../data/heuristics.csv', index=False)
