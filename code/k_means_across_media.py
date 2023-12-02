import time
import pandas as pd
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

print('starting k_means_across_media.py')
start_time = time.time()

#import the real dialogue data
dialogue = pd.read_csv('../data/dialogue.csv')
dialogue = dialogue.dropna(subset=['dialogue'])
#filter on column is_top_speaker is true
dialogue = dialogue[dialogue['is_top_six'] == True]

def prepare_data(df):    
    #init a default dict to store the data
    speaker_dict = defaultdict(list)
    for speaker in df['speaker'].unique():
        if len(df[df['speaker'] == speaker]['media'].unique()) > 1:
            print('this speaker is in more than one media')
            return None
        else:
            #filter the df to rows that contain the speaker
            speaker_df = df[df['speaker'] == speaker]
            #convert all lines of dialogue for the speaker into a list
            speaker_dialogue = speaker_df['dialogue'].tolist()
            #convert the list of speaker dialogue into a single string
            speaker_dialogue = ' '.join(speaker_dialogue)
            #append the speaker and dialogue to the speaker_dict
            speaker_dict[speaker] = speaker_dialogue
    return speaker_dict

def transform_data(speaker_dictionary):
    #create a pipeline for tfidf and svd
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), max_df=0.99),
        TruncatedSVD(n_components=2)
    )
    transformed_data = model.fit_transform(speaker_dictionary.values())
    return transformed_data

def plot_data(matrix):
    #create a scatterplot of the data and save as png
    plt.cla()
    plt.scatter(matrix[:, 0], matrix[:, 1], cmap='viridis')
    plt.savefig('../images/k_means_across_media.png')

    return None

print('building dictionary')
speaker_dialogues = prepare_data(dialogue)
print('performing tfidf and svd')
transformed_data = transform_data(speaker_dialogues)
print('plotting data')
plot_data(transformed_data)
end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")
