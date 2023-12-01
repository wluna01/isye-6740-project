import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import time

#start a timer
start_time = time.time()

#set random state
np.random.seed(6740)
random_state = 6740

#import the real dialogue data
dialogue = pd.read_csv('../data/dialogue.csv')
dialogue = dialogue.dropna(subset=['dialogue'])
#filter on column is_top_speaker is true
dialogue = dialogue[dialogue['is_top_six'] == True]

#convert the dialogue column to a list
dialogue_list = dialogue['dialogue'].tolist()
#convert the media column to a list
media_list = dialogue['media'].tolist()

#take 1000 samples from the dialogue_list and media_list for each media
#dialogue_list = dialogue.groupby('media').apply(lambda x: x.sample(n=1000)).reset_index(drop=True)['dialogue'].tolist()
#media_list = dialogue.groupby('media').apply(lambda x: x.sample(n=1000)).reset_index(drop=True)['media'].tolist()
#take 1000 samples from the dialogue_list and media_list
#create a mask of 1000 random rows

#use the mask to select 1000 random rows from the dialogue_list and media_list
#dialogue_list = np.array(dialogue_list)[mask].tolist()
#media_list = np.array(media_list)[mask].tolist()

#sample 10000 rows of each media from the dialogue_list and media_list
#for each media
#sample 10000 rows
#append to a new list
#dialogue_list = []
#media_list = []
#sample_size = 1000

#for media in dialogue['media'].unique():
#    media_df = dialogue[dialogue['media'] == media]
#    mask = np.random.randint(0, len(media_df), sample_size)
#    media_df = media_df.iloc[mask]
#    dialogue_list.append(media_df['dialogue'].tolist())
#    media_list.append(media_df['media'].tolist())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dialogue_list, media_list, test_size=0.2, random_state=random_state)

# Create a pipeline that first creates TF-IDF features and then trains a classifier
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 3), max_df=0.99),
    SMOTE(random_state=random_state),
    MultinomialNB()
)

# Train the model
model.fit(X_train, y_train)
# Predict genres for the test set
predicted_genres = model.predict(X_test)
# Assuming you have your test set and model ready
predicted_genres = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, predicted_genres, labels=model.classes_)
# Calculate class-specific accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
# Print class-specific accuracy
#for genre, accuracy in zip(model.classes_, class_accuracies):
#    print(f"Accuracy for {genre}: {accuracy}")

# print the mean of the class accuracies
print(f"Mean Genre Class accuracy: {np.mean(class_accuracies)}")

#end the timer
end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")
