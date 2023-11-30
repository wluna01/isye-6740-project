
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#seed the random state
np.random.seed(6740)

df = pd.read_csv('../data/dialogue.csv')
df = df.dropna()
df = df.sample(10000)

X = df[['avg_words_per_sentence', 'avg_chars_per_word']].values
y = df['media'].values

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6740)

#create kde for each media
kdes = {}
for media in np.unique(y_train):
    print('training classifier for ' + media)
    kde = KernelDensity(bandwidth=1)
    kde.fit(X_train[y_train == media])
    kdes[media] = kde

# Prediction function
def predict(X):
    scores = {genre: kde.score_samples(X) for genre, kde in kdes.items()}
    scores_df = pd.DataFrame(scores)
    return scores_df.idxmax(axis=1)

print('running prediction function')
# Predicting genres for the test set
y_pred = predict(X_test)

# Evaluating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
