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

# Targets
cat = preprocessing.LabelEncoder().fit(df['Species'])
y = tf.keras.utils.to_categorical(cat.transform(df['Species']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    shuffle=True, random_state=1)

"""
METRICS
"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sb

# load model
model = keras.models.load_model('trained models/original')
yp = model.predict(X_test)
pred_labels = np.argmax(yp, axis=1)
true_labels = np.argmax(y_test, axis=1)

print('true-labels: \n', true_labels, '\npred-labels: \n', pred_labels)

# confucsion matrix
cm = confusion_matrix(true_labels, pred_labels)
labels = df.Species.unique()
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

pp.figure()
sb.heatmap(df_cm, annot=True)
pp.title('Without standardising inputs \nAccuracy:{0:.3f}'.format(accuracy_score(true_labels, pred_labels)))
pp.ylabel('True label')
pp.xlabel('Predicted label')
pp.show()