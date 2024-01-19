import pandas as pd
import matplotlib.pyplot as pp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import numpy as np

"""
Compares the accuracy of the NN and other ML algorithms predictions
"""

# read data
df = pd.read_csv('iris.csv')

name='original'

# Inputs
xdf = df.drop(['Species'], axis=1)
if name == 'dim_reduction':
    xdf.drop('Petal.Length', axis=1, inplace=True)
    X = xdf.to_numpy()

elif name == 'standardised':
    xdf.drop('Petal.Length', axis=1, inplace=True)
    ss = preprocessing.StandardScaler()
    X = ss.fit_transform(xdf)

else:   # Original inputs
    X = xdf.to_numpy()

# Targets
cat = preprocessing.LabelEncoder().fit(df['Species'])
y = tf.keras.utils.to_categorical(cat.transform(df['Species']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    shuffle=True, random_state=1)

true_labels = np.argmax(y_test, axis=1)


"""
RANDOM FOREST
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sb

model = RandomForestClassifier(n_estimators = 10, random_state = 42)

# Train the model on training data
model.fit(X_train, y_train)

# Test on testing data. 
y_rf = model.predict(X_test)        # returns allocations

# need to transform allocations to categorical variable instances
# for the confusion matrix function
rf_labels = np.argmax(y_rf, axis=1)

# confucsion matrix
cm = confusion_matrix(true_labels, rf_labels)
labels = df.Species.unique()
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

# accuracy
rf_acc = metrics.accuracy_score(true_labels, rf_labels)

pp.figure()
sb.heatmap(df_cm, annot=True)
pp.title('Randon Forest \n Accuracy: {0:.3f}'.format(rf_acc))
pp.ylabel('True label')
pp.xlabel('Predicted label')

from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(model)

print('\nVariables that are affecting the product the most \n')
print(xdf.columns[sel.get_support()])


"""
Artificial Neural Network
"""

# load model
model = keras.models.load_model('trained models/' + name)
yp = model.predict(X_test)
pred_labels = np.argmax(yp, axis=1)
true_labels = np.argmax(y_test, axis=1)

# confucsion matrix
cm = confusion_matrix(true_labels, pred_labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

nn_acc = metrics.accuracy_score(true_labels, pred_labels)

pp.figure()
sb.heatmap(df_cm, annot=True)
pp.title(name + ' NN: Accuracy:{0:.3f}'.format(nn_acc))
pp.ylabel('True label')
pp.xlabel('Predicted label')
pp.show()