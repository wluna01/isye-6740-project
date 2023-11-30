import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#nltk.download('punkt')
dialogue = pd.read_csv('../data/dialogue.csv')
dialogue = dialogue.dropna(subset=['dialogue'])

import seaborn as sns
import matplotlib.pyplot as plt

#create a jointgrid

#create subset of dialogue df where media is The Simpsons
data1 = dialogue[dialogue['media'] == 'Pride and Prejudice']
data1 = data1.sample(1000)

g = sns.JointGrid(data=data1, x='avg_words_per_sentence', y='avg_chars_per_word', space=0)

#plot the first dataset's KDE
g = g.plot_joint(sns.kdeplot, data=data1, color="blue", alpha=0.2)

sns.kdeplot(data=data1['avg_words_per_sentence'], ax=g.ax_marg_x, color="blue", bw_adjust=0.5)
sns.kdeplot(data=data1['avg_chars_per_word'], ax=g.ax_marg_y, vertical=True, color="blue", bw_adjust=0.5)

# Add a legend to distinguish between the datasets
#plt.legend(labels=['Class 0', 'Class 1'])
#reduce the coverage of the axes to twenty by twenty
#g.ax_joint.set_xlim(0, 40)
#g.ax_joint.set_ylim(0, 12)

#save the plot as a png
plt.savefig('../images/3d_kde_test_0.png')

#create subset of dialogue df where media is Friends
data2 = dialogue[dialogue['media'] == 'Friends']
data2 = data2.sample(1000)

plt.cla()
g = sns.JointGrid(data=data2, x='avg_words_per_sentence', y='avg_chars_per_word', space=0)
g = g.plot_joint(sns.kdeplot, data=data2, color="red", alpha=0.2)

# Add the marginal univariate plots for each dataset
sns.kdeplot(data=data2['avg_words_per_sentence'], ax=g.ax_marg_x, color="red", bw_adjust=0.5)
sns.kdeplot(data=data2['avg_chars_per_word'], ax=g.ax_marg_y, vertical=True, color="red", bw_adjust=0.5)
#g.ax_joint.set_xlim(0, 40)
#g.ax_joint.set_ylim(0, 12)
plt.savefig('../images/3d_kde_test_1.png')

plt.cla()
##create subset of dialogue df where media is either The Simpsons or Pride and Prejudice
#data3 = dialogue[(dialogue['media'] == 'The Simpsons') | (dialogue['media'] == 'Pride and Prejudice')]

#c = sns.JointGrid(data=data3, x='avg_words_per_sentence', y='avg_chars_per_word', space=0, hue='media')
#c = c.plot_joint(sns.kdeplot, data=data3, alpha=0.2)

#sns.kdeplot(data=data3['avg_words_per_sentence'], ax=c.ax_marg_x, color="blue", bw_adjust=0.5)
#sns.kdeplot(data=data3['avg_chars_per_word'], ax=c.ax_marg_y, vertical=True, color="blue", bw_adjust=0.5)

#a = sns.JointGrid(data=data3, x='avg_words_per_sentence', y='avg_chars_per_word', space=0)
#a = a.plot_joint(sns.kdeplot, data=data2, color="red", alpha=0.2)
#sns.kdeplot(data=data2['avg_words_per_sentence'], ax=a.ax_marg_x, color="red", bw_adjust=0.5)
#sns.kdeplot(data=data2['avg_chars_per_word'], ax=a.ax_marg_y, vertical=True, color="red", bw_adjust=0.5)

#b = sns.JointGrid(data=data1, x='avg_words_per_sentence', y='avg_chars_per_word', space=0)
#b = b.plot_joint(sns.kdeplot, data=data1, color="blue", alpha=0.2)
#sns.kdeplot(data=data1['avg_words_per_sentence'], ax=b.ax_marg_x, color="blue", bw_adjust=0.5)
#sns.kdeplot(data=data1['avg_chars_per_word'], ax=b.ax_marg_y, vertical=True, color="blue", bw_adjust=0.5)
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_xlim(0, 40)
ax.set_ylim(0, 15)
sns.kdeplot(data=data1, x='avg_words_per_sentence', y='avg_chars_per_word', color="blue", alpha=0.5, ax=ax, bw_adjust=1)
sns.kdeplot(data=data2, x='avg_words_per_sentence', y='avg_chars_per_word', color="red", alpha=0.5, ax=ax, bw_adjust=1)


plt.savefig('../images/3d_kde_test_final.png')
