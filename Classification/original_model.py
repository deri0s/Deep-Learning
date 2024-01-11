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
# print(df.head(5))

#check for nan and null values
# print(pd.isnull(df).sum())

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
TRAIN AND TEST DATA
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
ARCHITECTURE
"""
from keras.layers import Dense as dense
import numpy as np

K = 3                               # Number of categories
N_train, D = np.shape(X_train)      # N: samples and dimensions

model = keras.models.Sequential()

model.add(dense(units=27, input_dim=D, activation='relu'))
model.add(dense(units=9, activation='relu'))
model.add(dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.summary()

"""
TRAINING

Define model "hyperparameters"
"""
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

B = int(N_train/1)
epoch = 300
val_split = 0.2

stop = EarlyStopping(monitor='val_accuracy', patience=50, mode='max')

trained = model.fit(X_train, y_train,
                    batch_size=B, epochs=epoch, validation_split=val_split,
                    verbose=1, callbacks=stop)

# Plot accuracy of the model after each epoch
pp.figure()
pd.DataFrame(trained.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy improvements with Epoch")

"""
PREDICTIONS
"""
print('\n Model accuracy \n')
model.evaluate(X_test, y_test)

# save model
model.save('original')

pp.show()