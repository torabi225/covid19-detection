# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:30:41 2021

@author: sara-pc

"""
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.utils import np_utils
import pandas as pd

#%% Section 36
dataset3 = pd.DataFrame({'Algorithms':['ourmodel','cnn','cnn+lstm'],'Accuracy':[89.22,87.35,62.59]})
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="Algorithms", y="Accuracy", data=dataset3 )

dataset4 = pd.DataFrame({'Algorithms':['ourmodel','cnn','cnn+lstm'],'precision':[85.20,82.42,55.06]})
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="Algorithms", y="precision", data=dataset4 )


dataset5 = pd.DataFrame({'Algorithms':['ourmodel','cnn','cnn+lstm'],'recall':[82.61,81.68,49.84]})
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="Algorithms", y="recall", data=dataset5 )

dataset6 = pd.DataFrame({'Algorithms':['ourmodel','cnn','cnn+lstm'],'f1score':[83.60,82.03,50.37]})
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="Algorithms", y="f1score", data=dataset6 )

