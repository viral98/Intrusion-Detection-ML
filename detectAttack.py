import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pylab


data_dir="./"
raw_data_filename = data_dir + "kddcup.data_10_percent"

print "Loading raw data"

raw_data = pd.read_csv(raw_data_filename, header=None)

print "Transforming data"
# Categorize columns: "protocol", "service", "flag", "attack_type"
raw_data[1], protocols= pd.factorize(raw_data[1])
raw_data[2], services = pd.factorize(raw_data[2])
raw_data[3], flags    = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

# separate features (columns 1..40) and label (column 41)
features= raw_data.iloc[:,:raw_data.shape[1]-1]
labels= raw_data.iloc[:,raw_data.shape[1]-1:]



# convert them into numpy arrays

labels= labels.values.ravel() # this becomes a 'horizontal' array

# Separate data in train set and test set
df= pd.DataFrame(features)
# create training and testing vars
# Note: train_size + test_size < 1.0 means we are subsampling
# Use small numbers for slow classifiers, as KNN, Radius, SVC,...
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print "X_train, y_train:", X_train.shape, y_train.shape
print "X_test, y_test:", X_test.shape, y_test.shape


print "Training model"
clf= RandomForestClassifier(n_jobs=-1, n_estimators=102, max_features=0.8, min_samples_leaf=3, n_estimators=500, min_samples_split=3, random_state=10, verbose=1)
#clf2 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, presort=False)

trained_model= clf.fit(X_train, y_train)

print "Score: ", trained_model.score(X_train, y_train)

# Predicting
print "Predicting"
y_pred = clf.predict(X_test)

print "Computing performance metrics"
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

sn.heatmap(results.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

plt.plot(X_test, y_test, 'b-')
# Plot the predicted values
plt.plot(X_test, y_pred, 'ro')
plt.xticks(rotation = '60')
plt.legend()
# Graph labels
plt.title('Actual and Predicted Values')
plt.show()


