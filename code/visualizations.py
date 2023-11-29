import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    #bring in data/heuristics.csv file as a dataframe
    heuristics = pd.read_csv('../data/heuristics.csv')
    #use MinMaxScaler to scale simpsons_d and yules_k to be between 0 and 1
    scaler = MinMaxScaler()
    heuristics[['simpsons_d', 'yules_k']] = scaler.fit_transform(heuristics[['simpsons_d', 'yules_k']])

    #create scatterplot of simpsons_d vs yules_k
    sns.scatterplot(x='simpsons_d', y='yules_k', hue='media', data=heuristics)
    plt.savefig('../images/heuristics.png')

    '''
    #create smaller df with only The Simpsons and Friends
    heuristics_simpsons_friends = heuristics[(heuristics['media'] == 'The Simpsons') | (heuristics['media'] == 'Friends')]
    #further filter by removing Mr. Burns and Mo
    heuristics_simpsons_friends = heuristics_simpsons_friends[~heuristics_simpsons_friends['speaker'].isin(['C. Montgomery Burns', 'Moe Szyslak'])]

    plt.cla()
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='simpsons_d', y='yules_k', hue='speaker', style='media', data=heuristics_simpsons_friends)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()
    plt.savefig('../images/heuristics_drilldown.png')
    '''
    #for each distinct value in the media column of the heuristics df, create a scatterplot of simpsons_d vs yules_k
    for media in heuristics['media'].unique():
        plt.cla()
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='simpsons_d', y='yules_k', hue='speaker', style='media', data=heuristics[heuristics['media'] == media])
        
        for i, txt in enumerate(heuristics[heuristics['media'] == media]['speaker']):
            plt.annotate(txt, (heuristics[heuristics['media'] == media]['simpsons_d'].iloc[i], heuristics[heuristics['media'] == media]['yules_k'].iloc[i]))
        #explicitly remove the legend
        plt.legend().remove()
        plt.tight_layout()
        #add the media as the title 
        plt.title(media)
        #add som extra space to the top of the plot to ensure the title is not cut off
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'../images/{media}_heuristics.png')
