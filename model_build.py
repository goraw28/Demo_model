import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\GauravUgale\\Desktop\\ML_projects\\all_signal_data1.csv")

df = data.drop(columns=['FileName'])
df.fillna( value= 0, inplace= True)
# print(df['Label'].value_counts())

# df['Label'] = df['Label'].replace({'2':3, '0':1, '--':0, '-':0, "5":2, "3":3, "1":1})
print(df['Label'].value_counts())

columns = ['mean', 'std_dev', 'kurtosis', 'median', 'zero_crossings', 'slope_changes',
'max_freq_amp', 'peak_freq', 'spectral_skewness', 'spectral_kurtosis', 'std', 
'correlate', 'mean_absolute_percentage_error', 'min_val', 'max_val', 'range_val', 'skewness', 'mode', 'spectral_centroid', 
'spectral_bandwidth', 'gradient_std', 'gradient_pVar', 'pVar', 'max_error', 
'd2_tweedie_score', 'd2_pinball_score', 'kurtosis.1', 'Label']

# drop_colums = ['min_val', 'max_val', 'range_val', 'skewness', 'mode', 'spectral_centroid', 
# 'spectral_bandwidth', 'gradient_std', 'gradient_pVar', 'pVar', 'max_error', 
# 'd2_tweedie_score', 'd2_pinball_score', 'kurtosis.1', 'Label']


# x = df.drop(columns=drop_colums)

print(df.columns)
plt.figure(figsize=(25,15))
sns.heatmap(df.corr(), annot=True)
plt.savefig('correlation1.jpg') 

# '''   

x = df.drop(columns='Label').values
x = df.values
y = df['Label'].values

print(x.shape, y.shape)

# models = {'LogisticRegression': LogisticRegression(),
#           'KNeighborsClassifier': KNeighborsClassifier(),
#           'DecisionTreeClassifier':DecisionTreeClassifier(),
#           'RandomForestClassifier': RandomForestClassifier(),
#           'GradientBoostingClassifier': GradientBoostingClassifier(),
#           'AdaBoostClassifier': AdaBoostClassifier(),
#           'ExtraTreesClassifier': ExtraTreesClassifier(),
#           'NaiveBayes': GaussianNB(),
#           'DummyClassifier': DummyClassifier(),
#           'SVM': SVC()}

models = {'DecisionTreeClassifier':DecisionTreeClassifier(),
          'RandomForestClassifier': RandomForestClassifier(),
          'ExtraTreesClassifier': ExtraTreesClassifier(),
          'SVM': SVC()}

L = 0
for _ in models.keys():
    if len(_) > L:
        L = len(_)

for name, model in models.items():
    model.fit(x, y)
    l = L-len(name)+2
    print(name, f'{"-"*l}>', model.score(x, y))
    y_pred = model.predict(x)
    
all_model = []

for name, model in models.items():
    a = (name, model)
    all_model.append(a)
    
stc = StackingClassifier(all_model, n_jobs=-1, verbose=True)
print(stc)
stc.fit(x, y)
print("\n\nAccuracy ::", stc.score(x, y))
y_pred = stc.predict(x)
print("\n\nClassification Report ::\n", classification_report(y, y_pred))


c = confusion_matrix(y, y_pred)
# c = np.insert(c, 0, values=[0, 1], axis=1)
# c = np.insert(c, 0, values=[0, 0, 1], axis=0)
print('\n\nConfusion Matrix ::\n', c)

pickle.dump(stc, open('stc_new_2(1).pkl', 'wb'))
# '''