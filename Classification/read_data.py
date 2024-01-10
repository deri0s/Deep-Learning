import pandas as pd
import matplotlib.pyplot as pp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

"""
PRE-PROCESSING
"""
# read data
df = pd.read_csv('iris.csv')
# print(df.head(5))

#check for nan and null values
# print(pd.isnull(df).sum())

# check input data
xdf = df.drop(['Species'], axis=1)
# print(xdf.head(3))
pp.title('Inputs')
pp.plot(df.index, xdf, label=list(xdf.columns.values))
pp.legend()

# convert output into a categorical variable
#                   string -> number            fir and predict
df['Species'] = preprocessing.LabelEncoder().fit_transform(df['Species'])
# converts numbers into categorical variable: 
y = tf.keras.utils.to_categorical(df['Species'].values)
print('y: ', y, '\n', type(y))

"""
TRAIN AND TEST DATA
"""

# pp.show()