import pandas as pd
import glob
import re

def df_to_dialogue(df, speaker_col, dialoge_col, media_name):
    df['speaker'] = df[speaker_col]
    df['dialogue'] = df[dialoge_col]
    df['media'] = media_name

    df = df[['speaker', 'dialogue', 'media']]
    return df

def concat_dfs(*dfs):
    df = pd.concat(dfs, ignore_index=True)
    return df

def prepare_txt_data(file_path):

    files = glob.glob(file_path)

    all_lines = []

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != '']
            all_lines.extend(lines)

    all_lines = [line.split(':') for line in all_lines]
    all_lines = [line for line in all_lines if len(line) == 2]

    df = pd.DataFrame(all_lines, columns=['speaker', 'dialogue'])
    return df

def prepare_unstructured_data(file_path):

    files = glob.glob(file_path)

    all_lines = []

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            all_lines.extend(lines)
    
    #remove the newline character from the end of each line
    all_lines = [line.strip() for line in all_lines]

    all_dialogue = []

    for i in range(len(all_lines)):

        #evaluate whether the line is a speaker or dialogue
        if all_lines[i].isupper() and i < len(all_lines):
            #add the current line and the next line to the dialogue list as a tuple
            all_dialogue.append((all_lines[i], all_lines[i+1]))

    #convert the dialogue list to a dataframe
    df = pd.DataFrame(all_dialogue, columns=['speaker', 'dialogue'])

    return df

def prepare_data():
    simpson = pd.read_csv('../data/simpsons/simpsons_dataset.csv')
    rickmorty = pd.read_csv('../data/rickandmorty/RickAndMortyScripts.csv')
    lotr = pd.read_csv('../data/lotr/lotr_scripts.csv')
    friends = prepare_txt_data('../data/friends/*.txt')
    prideprejudice = prepare_txt_data('../data/prideprejudice/*.txt')
    downtownabbey = prepare_unstructured_data('../data/downtownabbey/*.txt')

    df = concat_dfs(
            df_to_dialogue(rickmorty, 'name', 'line', 'Rick and Morty'),
            df_to_dialogue(lotr, 'char', 'dialog', 'The Lord of the Rings'),
            df_to_dialogue(simpson, 'raw_character_text', 'spoken_words', 'The Simpsons'),
            df_to_dialogue(friends, 'speaker', 'dialogue', 'Friends'),
            df_to_dialogue(prideprejudice, 'speaker', 'dialogue', 'Pride and Prejudice'),
            df_to_dialogue(downtownabbey, 'speaker', 'dialogue', 'Downtown Abbey')
        )
    
    #apply title case to the speaker column
    df['speaker'] = df['speaker'].str.title()

    #trim quotation marks and spaces from the dialogue
    #df['dialogue'] = df['dialogue'].str.strip().str.strip('\"')
    #df['dialogue'] = df['dialogue'].apply(lambda text: re.sub(r'["“”]', '', str(text)))
    cleaned_df = df.dropna()
    
    return cleaned_df

df = prepare_data()
#remove all rows where the dialogue column does not contain a string
df = df[df['dialogue'].apply(lambda text: isinstance(text, str))]
df = df.dropna()

print(df.groupby('media')['speaker'].value_counts().groupby(level=0).head(5))
#save the dataframe to a csv file
df.to_csv('../data/dialogue.csv', index=False)
