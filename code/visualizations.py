import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#bring in data/heuristics.csv file as a dataframe
heuristics = pd.read_csv('../data/heuristics.csv')
#print(heuristics.head())
#create scatterplot of simpsons_d vs yules_k
sns.scatterplot(x='simpsons_d', y='yules_k', hue='media', data=heuristics)
#save the scatterplot as a png
plt.savefig('../images/heuristics.png')



