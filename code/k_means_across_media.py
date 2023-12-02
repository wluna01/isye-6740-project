import time
import pandas as pd
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches

print('starting k_means_across_media.py')
start_time = time.time()

#import the real dialogue data
dialogue = pd.read_csv('../data/dialogue.csv')
dialogue = dialogue.dropna(subset=['dialogue'])
#filter on column is_top_speaker is true
dialogue = dialogue[dialogue['is_top_six'] == True]

def prepare_data(df):    
    #init a default dict to store the data
    speaker_dict = defaultdict(dict)
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
            speaker_dict[speaker]['dialogue'] = speaker_dialogue
            speaker_dict[speaker]['media'] = speaker_df['media'].unique()[0]
    speaker_df = pd.DataFrame.from_dict(speaker_dict, orient='index').reset_index()
    return speaker_df

def transform_data(speaker_df):
    #create a pipeline for tfidf and svd
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), max_df=0.99),
        TruncatedSVD(n_components=2)
    )
    transformed_data = model.fit_transform(speaker_df['dialogue'].tolist())
    
    return transformed_data

def plot_data(matrix, speakers, media, response_type):
    #create a scatterplot of the data and save as png
    plt.cla()

    if response_type == 'actual':
    #create a dictionary of colors for each media
        labels = {'The Simpsons': 'red', 'Friends': 'blue', 'Downtown Abbey': 'green', 'The Lord of the Rings': 'orange', 'Rick and Morty': 'purple', 'Pride and Prejudice': 'yellow'}
    elif response_type == 'predicted':
        #create a dictionary of colors for each media
        labels = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple', 5: 'yellow'}

    for i, txt in enumerate(speakers):
        plt.scatter(matrix[i, 0], matrix[i, 1], color=labels[media[i]])
        plt.annotate(txt, (matrix[:, 0][i], matrix[:, 1][i]))
    plt.savefig(f'../images/k_means_across_media_{response_type}.png')
    return None

def cluster_data(matrix):
    #run kmeans clustering
    kmeans = KMeans(n_clusters=6, random_state=6740, n_init=20, max_iter=1000)
    kmeans.fit(matrix)
    return kmeans.labels_

def get_purity_score(speaker, actual, predicted):
    #create a df with the actual and predicted labels
    df = pd.DataFrame({'speaker': speaker, 'actual': actual, 'predicted': predicted})
    #determine the the most common actual label for each predicted label
    outcome_df = df.groupby('predicted').agg(lambda x:x.value_counts().index[0])
    #add a column to the original df of predicted category based on the outcome_df
    df['predicted_category'] = df['predicted'].map(outcome_df['actual'])
    #add a column that's true if the actual and predicted category are the same
    df['is_correct'] = df['actual'] == df['predicted_category']
    print(df)

    return None

def plot_data_selective_annotation(speakers, media, matrix):
    #create a scatterplot of the data and save as png
    plt.cla()
    legend_handles = []
    labels = {'The Simpsons': 'red', 'Friends': 'blue', 'Downtown Abbey': 'green', 'The Lord of the Rings': 'orange', 'Rick and Morty': 'purple', 'Pride and Prejudice': 'yellow'}
    for i, txt in enumerate(speakers):
        plt.scatter(matrix[i, 0], matrix[i, 1], color=labels[media[i]])

    patches = [mpatches.Patch(color=color, label=label) for label, color in labels.items()]
    plt.legend(handles=patches)
    #show the legend on the plot
    plt.savefig(f'../images/k_means_across_media_misclassifications.png')
    return None

print('building dictionary')
speaker_df = prepare_data(dialogue)

print('performing tfidf and svd')
transformed_data = transform_data(speaker_df)

print('clustering data')
cluster_labels = cluster_data(transformed_data)
#add the cluster labels to the speaker_df
speaker_df['cluster_label'] = cluster_labels

print('plotting data with media labels')
#plot_data(transformed_data, speaker_df['index'].tolist(), speaker_df['media'].tolist(), 'actual')

print('plotting data with cluster labels')
#plot_data(transformed_data, speaker_df['index'].tolist(), speaker_df['cluster_label'].tolist(), 'predicted')

print('finding purity score')
get_purity_score(speaker_df['index'].tolist(), speaker_df['media'].tolist(), speaker_df['cluster_label'].tolist())

print('plotting final chart with selective annotation')
plot_data_selective_annotation(speaker_df['index'].tolist(), speaker_df['media'].tolist(), transformed_data)

end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")
