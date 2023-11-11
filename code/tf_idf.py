import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the data
dialogue = pd.read_csv('../data/dialogue.csv')

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

#include bigrams and trigrams, remove words spoken by all family members
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.99)

#fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(dialogue)
#get feature names (words or n-grams)
feature_names = vectorizer.get_feature_names_out()

#analyze TF-IDF scores per document
for doc_index, doc in enumerate(tfidf_matrix):
    print(f"Document {doc_index}:")
    
    # Convert the TF-IDF matrix for this document into a readable format
    df = pd.DataFrame(doc.T.todense(), index=feature_names, columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    
    # Display the top terms with their scores
    print(df.head(10))  # Display top 10 terms for this document

# Show the TF-IDF matrix
print(tfidf_matrix.toarray().shape)

cosine_similarities = cosine_similarity(tfidf_matrix)
print(cosine_similarities)
