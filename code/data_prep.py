import pandas as pd

def df_to_dialogue(df, speaker_col, dialoge_col, media_name):
    df['speaker'] = df[speaker_col]
    df['dialoge'] = df[dialoge_col]
    df['media'] = media_name

    df = df[['speaker', 'dialoge', 'media']]
    return df

def concat_dfs(*dfs):
    df = pd.concat(dfs, ignore_index=True)
    return df

simpson = pd.read_csv('../data/simpsons/simpsons_dataset.csv')
rickmorty = pd.read_csv('../data/rickandmorty/RickAndMortyScripts.csv')
lotr = pd.read_csv('../data/lotr/lotr_scripts.csv')

df = concat_dfs(df_to_dialogue(rickmorty, 'name', 'line', 'Rick and Morty'), df_to_dialogue(lotr, 'char', 'dialog', 'The Lord of the Rings'), df_to_dialogue(simpson, 'raw_character_text', 'spoken_words', 'The Simpsons'))

print(df.groupby('media')['speaker'].value_counts().groupby(level=0).head(5))
