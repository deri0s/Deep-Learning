import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

df = pd.read_csv('root_cause_analysis.csv')
# print(df.head(5))

# print(df.isnull().sum())                              # It does not have null values
# print(df.ROOT_CAUSE.value_counts())                   # It is not an unbalanced problem

# print('Should I standardise the inputs?')
# for cols in df.columns.values:
#     print(cols, ' ', np.unique(df[cols].values))      # There is no need to standardise the inputs

"""
Preprocess
"""
from tensorflow import keras
from keras import utils
from sklearn.model_selection import train_test_split

xdf = df.drop(columns=['ID', 'ROOT_CAUSE'], axis=1)
N, D = np.shape(xdf)

encoder = preprocessing.LabelEncoder()
y_labels = encoder.fit_transform(df.ROOT_CAUSE.values)
y_cat = utils.to_categorical(y_labels)
print('N: ', N, ' D: ', D)

# X, y
X = xdf.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2,
                                                    shuffle=True, random_state=1,)

"""
ARCHITECTURE
"""
# from keras.layers import Dense as dense

# model = keras.models.Sequential()

# model.add(dense(units=64, input_dim=D, activation='relu'))
# model.add(dense(units=8, activation='relu'))
# model.add(dense(units=3, activation='softmax'))

# model.compile(loss='categorical_crossentropy', metrics='accuracy')
# model.summary()

"""
TRAINING
"""
import matplotlib.pyplot as plt
# from keras.callbacks import EarlyStopping

# B = int(N/2)
# epoch = 500
# val_split = 0.2

# stop = EarlyStopping(monitor='val_accuracy', patience=100, mode='max')

# hist = model.fit(X_train, y_train,
#                  batch_size=B, epochs=epoch, validation_split=val_split,
#                  verbose=1, callbacks=stop)

# # Plot accuracy for each epoch
# plt.figure()
# plt.plot(pd.DataFrame(hist.history)['accuracy'])

"""
Predictions
"""
# print('Model accuracy \n')
# model.evaluate(X_test, y_test)

# model.save('root')

"""
METRICS
"""
import seaborn as sb
from sklearn.metrics import confusion_matrix as cm

model = keras.models.load_model('root')
pred_labels = np.argmax(model.predict(X_test), axis=1)
true_labels = np.argmax(y_test, axis=1)

m = cm(true_labels, pred_labels)
labels = df.ROOT_CAUSE.unique()

plt.figure()
sb.heatmap(pd.DataFrame(m, index=labels, columns=labels),
           annot=True)
plt.title('Model accuracy')
plt.xlabel('Predicted labels')
plt.xlabel('True labels')
plt.show()

"""
Pass individual sample
"""
print('what is tf.expand_dims? \n', tf.expand_dims(X_test[-1], axis=0))
pred = np.argmax(model.predict(tf.expand_dims(X_test[-1], axis=0))[0])
print('Predicted cause: ', encoder.inverse_transform([pred]),
      ' true cause: ', encoder.inverse_transform([true_labels[-1]]))