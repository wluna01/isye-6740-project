import pandas as pd
#import dialogue.csv
dialogue = pd.read_csv('../data/dialogue.csv')
#print ten distinct speakers where is_top_speaker is false
print(dialogue[dialogue['is_top_six'] == False].sample(10))
#print ten distinct speakers where is_top_speaker is true
print(dialogue[dialogue['is_top_six'] == True].sample(10))
