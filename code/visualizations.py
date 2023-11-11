import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#bring in data/heuristics.csv file as a dataframe
heuristics = pd.read_csv('../data/heuristics.csv')

#create scatterplot of simpsons_d vs yules_k
sns.scatterplot(x='simpsons_d', y='yules_k', hue='media', data=heuristics)
plt.savefig('../images/heuristics.png')

#create smaller df with only The Simpsons and Friends
heuristics_simpsons_friends = heuristics[(heuristics['media'] == 'The Simpsons') | (heuristics['media'] == 'Friends')]
#further filter by removing Mr. Burns and Mo
heuristics_simpsons_friends = heuristics_simpsons_friends[~heuristics_simpsons_friends['speaker'].isin(['C. Montgomery Burns', 'Moe Szyslak'])]

plt.cla()
plt.figure(figsize=(8, 8))
sns.scatterplot(x='simpsons_d', y='yules_k', hue='speaker', style='media', data=heuristics_simpsons_friends)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.savefig('../images/heuristics_drilldown.png')

