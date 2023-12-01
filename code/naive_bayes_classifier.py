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

#convert the dialogue column to a list
dialogue_list = dialogue['dialogue'].tolist()
#convert the media column to a list
media_list = dialogue['media'].tolist()

# Sample data (Replace this with your dataset)
#dialogs = ['I will be back', 'May the force be with you', 'You talking to me?', 'Elementary, my dear Watson']
#genres = ['Action', 'Sci-fi', 'Drama', 'Mystery']

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(dialogs, genres, test_size=0.2, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(dialogue_list, media_list, test_size=0.2, random_state=random_state)

# Create a pipeline that first creates TF-IDF features and then trains a classifier
model = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 3)),
        SMOTE(random_state=random_state),
        MultinomialNB
    ())

# Train the model
model.fit(X_train, y_train)

# Predict genres for the test set
predicted_genres = model.predict(X_test)

# Evaluate the model
#print(classification_report(y_test, predicted_genres))

#get the accuracy of the model overall
#print(model.score(X_test, y_test))

# Assuming you have your test set and model ready
predicted_genres = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, predicted_genres, labels=model.classes_)

# Calculate class-specific accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)

# Print class-specific accuracy
for genre, accuracy in zip(model.classes_, class_accuracies):
    print(f"Accuracy for {genre}: {accuracy}")

#end the timer
end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")
