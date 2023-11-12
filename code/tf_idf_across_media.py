import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#load the data
dialogue = pd.read_csv('../data/dialogue.csv')

def prepare_document_corpus(corpus):
    
    document_list = []
    
    #get the unique entries in the media column of the corpus df
    media = corpus['media'].unique().tolist()
    for each in media:
        #filter the corpus df by the current media
        media_df = corpus[corpus['media'] == each]
        #convert the dialogue to a list
        media_dialogue = media_df['dialogue'].dropna().tolist()
        #convert the list to a string
        media_dialogue = ' '.join(media_dialogue)
        document_list.append(media_dialogue)

    return document_list

dialogue_list = prepare_document_corpus(dialogue)
#include bigrams and trigrams, remove words spoken by all family members
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.99)

#fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(dialogue_list)
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

pca = PCA(n_components=2)
dense_tfidf_matrix = np.asarray(tfidf_matrix.todense())
reduced_data = pca.fit_transform(dense_tfidf_matrix)

# Plot
plt.cla()
plt.figure(figsize=(10, 10))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
for i, text in enumerate(reduced_data):
    plt.annotate(f'Doc {i}', (text[0], text[1]))
plt.title("TF-IDF Document Similarity - Across Media")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig('../images/tfidf_across_media.png')