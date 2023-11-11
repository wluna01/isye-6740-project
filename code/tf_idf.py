import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the data
dialogue = pd.read_csv('../data/dialogue.csv')

#return to this 
#get all unique media
#media = dialogue['media'].unique().tolist()
#print(media)

#select only dialogue from The Simpsons
simpsons = dialogue[dialogue['media'] == 'The Simpsons']
simpsons_family = ['Homer Simpson', 'Marge Simpson', 'Bart Simpson', 'Lisa Simpson']
dialogue = []

for family_member in simpsons_family:
    #get all dialogue from the family member
    family_member_df = simpsons[simpsons['speaker'] == family_member]
    #convert the dialogue to a list
    family_member_dialogue = family_member_df['dialogue'].tolist()
    #convert the list to a string
    family_member_dialogue = ' '.join(family_member_dialogue)
    dialogue.append(family_member_dialogue)

#print(dialogue[1])

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(dialogue)

# Show the TF-IDF matrix
print(tfidf_matrix.toarray().shape)

cosine_similarities = cosine_similarity(tfidf_matrix)
print(cosine_similarities)