import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

read_df = pd.read_csv('BreastCanser.csv', encoding='unicode_escape')
df = read_df.copy()



label_encoder = preprocessing.LabelEncoder()
df.diagnosis = label_encoder.fit_transform(df.diagnosis)
sns.set(style = 'darkgrid')
sns.lmplot(x = 'area_mean', y = 'diagnosis', data = df, height = 10, aspect = 1.5, y_jitter = 0.1)
plt.show()