import pandas as pd
import matplotlib.pyplot as pp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

"""
PRE-PROCESSING
"""
# read data
df = pd.read_csv('iris.csv')

# Inputs
xdf = df.drop(['Species'], axis=1)
X = xdf.to_numpy()

pp.figure()
pp.title('Inputs')
pp.plot(df.index, xdf, label=list(xdf.columns.values))
pp.legend()

print(xdf.corr())

# Targets
# convert output into a categorical variable
#                   string -> number            fit and predict
df['Species'] = preprocessing.LabelEncoder().fit_transform(df['Species'])
# converts numbers into categorical variable: 
y = tf.keras.utils.to_categorical(df['Species'].values)

"""
LOAD ORIGINAL NN
"""
model = keras.models.load_model('original')
result = model.predict(X)
print('what is result?\n', result)
# pp.show()