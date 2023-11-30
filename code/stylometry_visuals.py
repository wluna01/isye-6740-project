import pandas as pd
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from collections import Counter
from stylometry_helper_functions import get_sentence_lengths

#nltk.download('punkt')
dialogue = pd.read_csv('../data/dialogue.csv')

#print(dialogue.head())
def get_speaker_sentence_lengths(dialogue):
    #for each value in the media column
    for media in dialogue['media'].unique():
        #create a df with only the rows where the media column is equal to the current media value
        media_df = dialogue[dialogue['media'] == media]
        
        #identify the top 6 speakers in the media_df
        top_speakers = media_df['speaker'].value_counts().index.tolist()[:6]
        #for the top speakers in the media_df
        plt.cla()
        plt.xlabel('Sentence Length (words)')
        plt.ylabel('Density')
        for speaker in top_speakers:
            #filter the media_df to only include the speaker
            speaker_df = media_df[media_df['speaker'] == speaker]
            #convert the dialogue column to a list of sentences
            #sentences = speaker_df['dialogue'].tolist()
            #use the get_sentence_lengths function to calculate the sentence length distribution
            #sentence_lengths = get_sentence_lengths(sentences)
            sentence_lengths = speaker_df['avg_words_per_sentence'].tolist()
            #plot the distribution
            sns.kdeplot(sentence_lengths, bw_adjust=0.8, label=speaker)
            
        plt.title(media)
        plt.legend(title='Speaker')
        plt.savefig(f'../images/{media}_sentence_length.png')

def get_media_sentence_lengths(dialogue):
    #do the same as above, comparing across media
    plt.cla()
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Density')

    for media in dialogue['media'].unique():
        #create a df with only the rows where the media column is equal to the current media value
        media_df = dialogue[dialogue['media'] == media]
        #identify the top 6 speakers in the media_df
        top_speakers = media_df['speaker'].value_counts().index.tolist()[:6]
        #filter the media_df to only include the top speakers
        media_df = media_df[media_df['speaker'].isin(top_speakers)]
        sentences = media_df['dialogue'].tolist()
        #use the get_sentence_lengths function to calculate the sentence length distribution
        sentence_lengths = media_df['avg_words_per_sentence'].tolist()
        #plot the distribution
        sns.kdeplot(sentence_lengths, bw_adjust=1.2, label=media)

    #add a title and legend
    plt.title('Across Media Distributions')
    plt.legend(title='Media')
    #restrict x axis to be between 0 and 40
    plt.xlim(0, 30)
    plt.savefig(f'../images/overall_sentence_length.png')

def get_media_word_lengths(dialogue):
    #do the same as above, comparing across media
    plt.cla()
    plt.xlabel('Average Word Length')
    plt.ylabel('Density')

    return None

#get_speaker_sentence_lengths(dialogue)
#get_media_sentence_lengths(dialogue)

get_media_word_lengths(dialogue)
