import pandas as pd
import matplotlib.pyplot as pp
import seaborn as sb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

"""
PRE-PROCESSING
"""
# read data
df = pd.read_csv('iris.csv')

# Inputs
xdf = df.drop(['Species'], axis=1)
X = xdf.to_numpy()

pp.figure()
pp.plot(df.index.values, xdf, label=xdf.columns.values)
pp.title('Inputs')
pp.xlabel('Index')
pp.ylabel('cm')
pp.legend()

pp.figure()
sb.heatmap(xdf.corr(), annot=True)
pp.title('Inputs Correlation Matrix')

# Targets
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# is this an unbablanced classification problem?
print('label counts \n', df.Species.value_counts())

# convert string labels into categorical variable
cat = preprocessing.LabelEncoder().fit(df['Species'])
y = tf.keras.utils.to_categorical(cat.transform(df['Species']))

# Target labels as numbers
df_num = df.copy()
df_num['Species'] = df_num.Species.astype('category').cat.codes

# creates columns where the value of each species is True when equal to the column
# species
df_num = pd.concat((df_num, pd.get_dummies(df_num.Species, prefix='sp')), axis=1)

pp.figure()
sb.heatmap(df_num.corr(), annot=True)
pp.title('Targets Correlation matrix')

"""
- From the correlation anaysis, we observe that Petal Length and Petal Width are highly
correlated

- Although Petal Width has a correlation of 0.96 with the Targets (Species), I decided to
eliminate Petal Width, in the dimensionality reduction analysis, because Petal Width is
highly correlated with the label `setosa`.

- Another reason for the above decision is that Petal Length has a 0.95 correlation with
the target Species.
"""

sb.pairplot(df, hue="Species")
pp.show()

"""
! Important

After training the model without the Petal.Width feature, I noticed that It was very difficult to obtained
accuracy values above 93%.

On the other hand when training the NN without the Petal.Length feature, it was easier to obtain accuracy
values around 96%.

Therefore, I conclude that although the `linear correlation` between the Petal variables with the targets
are very similar. There might be `non-liear correlations between the 3 Sepal variables that facilitates
clustering.

Note that this correlations cannot be seen in the seabor.pairplot() analysis.
"""