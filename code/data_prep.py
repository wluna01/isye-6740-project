import pandas as pd
import glob

def df_to_dialogue(df, speaker_col, dialoge_col, media_name):
    df['speaker'] = df[speaker_col]
    df['dialoge'] = df[dialoge_col]
    df['media'] = media_name

    df = df[['speaker', 'dialoge', 'media']]
    return df

def concat_dfs(*dfs):
    df = pd.concat(dfs, ignore_index=True)
    return df

def prepare_friends_data():

    files = glob.glob('../data/friends/*.txt')

    all_lines = []

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != '']
            all_lines.extend(lines)

    all_lines = [line.split(':') for line in all_lines]
    all_lines = [line for line in all_lines if len(line) == 2]
    all_lines = [line for line in all_lines if line[0] in ['Joey', 'Chandler', 'Monica', 'Rachel', 'Ross', 'Phoebe']]

    df = pd.DataFrame(all_lines, columns=['speaker', 'dialogue'])
    return df

def prepare_data():
    simpson = pd.read_csv('../data/simpsons/simpsons_dataset.csv')
    rickmorty = pd.read_csv('../data/rickandmorty/RickAndMortyScripts.csv')
    lotr = pd.read_csv('../data/lotr/lotr_scripts.csv')
    friends = prepare_friends_data()

    df = concat_dfs(
            df_to_dialogue(rickmorty, 'name', 'line', 'Rick and Morty'),
            df_to_dialogue(lotr, 'char', 'dialog', 'The Lord of the Rings'),
            df_to_dialogue(simpson, 'raw_character_text', 'spoken_words', 'The Simpsons'),
            df_to_dialogue(friends, 'speaker', 'dialogue', 'Friends')
        )
    return df

df = prepare_data()
print(df.groupby('media')['speaker'].value_counts().groupby(level=0).head(5))
