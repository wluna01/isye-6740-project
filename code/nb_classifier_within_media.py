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

print('hello world')

#start a timer
start_time = time.time()

#set random state
random_state = 6740
np.random.seed(random_state)



#convert the dialogue column to a list
#dialogue_list = dialogue['dialogue'].tolist()
#convert the media column to a list
#speaker_list = dialogue['speaker'].tolist()

def run_within_media_nb_classifier(dialogue, response):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dialogue, response, test_size=0.2, random_state=random_state)

    # Create a pipeline that first creates TF-IDF features and then trains a classifier
    model = make_pipeline(
        TfidfVectorizer(vocabulary=None, ngram_range=(1, 3), max_df=0.99),
        SMOTE(random_state=random_state),
        MultinomialNB()
    )

    # Train the model
    model.fit(X_train, y_train)
    # Predict speakers for the test set
    predicted_speakers = model.predict(X_test)
    # Assuming you have your test set and model ready
    predicted_speakers = model.predict(X_test)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, predicted_speakers, labels=model.classes_)
    media_accuracy = cm.diagonal().sum() / cm.sum()
    #print overall accuracy (not sure this helps)
    #print(f"Media accuracy: {media_accuracy}")
    # Calculate class-specific accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    # Print class-specific accuracy
    #for speaker, accuracy in zip(model.classes_, class_accuracies):
    #    print(f"Accuracy for {speaker}: {accuracy}")
    # print the mean of the class accuracies
    print(f"Mean Speaker Class accuracy: {np.mean(class_accuracies)}")
    return np.mean(class_accuracies)

#import the real dialogue data
dialogue = pd.read_csv('../data/dialogue.csv')
dialogue = dialogue.dropna(subset=['dialogue'])

accuracies = []
#run the classifier for each media
for media in dialogue['media'].unique():
    print(f"Running classifier for {media}")
    #filter the dialogue df by the current media
    dialogue_within_media = dialogue[dialogue['media'] == media]
    #get the top six speakers
    top_speakers = dialogue_within_media['speaker'].value_counts().head(6).index.tolist()
    #filter the dialogue df by the top six speakers
    dialogue_within_media = dialogue_within_media[dialogue_within_media['speaker'].isin(top_speakers)]
    #convert the dialogue column to a list
    dialogue_list = dialogue_within_media['dialogue'].tolist()
    #convert the speaker column to a list
    speaker_list = dialogue_within_media['speaker'].tolist()

    #pull up to a thousand data points from the list
    dialogue_list = dialogue_list[:10000]
    speaker_list = speaker_list[:10000]

    #run the classifier
    accuracy = run_within_media_nb_classifier(dialogue_list, speaker_list)
    accuracies.append(accuracy)

#end the timer
end_time = time.time()
print(f"Runtime of the program is {end_time - start_time}")
#print the mean of the accuracies
print(f"Mean Overall Class Accuracy: {np.mean(accuracies)}")
