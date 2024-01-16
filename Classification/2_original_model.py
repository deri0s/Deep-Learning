import numpy as np
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

name='original'

# Inputs
xdf = df.drop(['Species'], axis=1)
if name == 'dim_reduction':
    xdf.drop('Petal.Length', axis=1, inplace=True)

xdf = df.drop(['Species'], axis=1)

# from 1_data_analysis we know that Petal width can be eliminated
xdf.drop('Petal.Length', axis=1, inplace=True)
X = xdf.to_numpy()

# Targets
# convert output into a categorical variable
#                   string -> number            fit and predict
df['Species'] = preprocessing.LabelEncoder().fit_transform(df['Species'])
# converts numbers into categorical variable: 
y = tf.keras.utils.to_categorical(df['Species'].values)


"""
TRANING AND TEST
"""
test_per = 0.2              # % of data used for model testing
N, D = np.shape(X)          # N: samples and dimensions

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_per,
                                                    shuffle=True, random_state=1)

N_train = len(y_train)


"""
ARCHITECTURE
"""
from keras.layers import Dense as dense
import numpy as np

K = 3                               # Number of categories

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

B = int(N_train/2)
epoch = 300
val_split = 0.2             # of the training data used for model 
                            # validation after each epoch

stop = EarlyStopping(monitor='val_accuracy', patience=100, mode='max')

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
# model.save('trained models/' + name)

pp.show()