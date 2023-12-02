import time
import pandas as pd
from collections import defaultdict

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

print('building dictionary')
speaker_dialogues = prepare_data(dialogue)




end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")
