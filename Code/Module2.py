#  Demonstration code developed bu Mayo Clinic Radiology Informatics Laboratory
#
#
# Module 2 - Descision Trees
# Load the libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.metrics as metrics
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

# read the CSV file

Data=pd.read_csv ('DataExample.csv')

"""

if you need to print or have access to the data as numpy array

print (Data)
print(Data.as_matrix(columns=['NAWMpost']))

"""

ClassNAWMpost=(Data['NAWMpost'].values)
ClassNAWMpost= (np.asarray(ClassNAWMpost))
ClassNAWMpost=ClassNAWMpost[~np.isnan(ClassNAWMpost)]
ClassNAWMpre=(Data[['NAWMpre']].values)
ClassNAWMpre= (np.asarray(ClassNAWMpre))
ClassNAWMpre=ClassNAWMpre[~np.isnan(ClassNAWMpre)]
ClassTUMORpost=(Data[['TUMORpost']].values)
ClassTUMORpost= (np.asarray(ClassTUMORpost))
ClassTUMORpost=ClassTUMORpost[~np.isnan(ClassTUMORpost)]
ClassTUMORpre=(Data[['TUMORpre']].values)
ClassTUMORpre= (np.asarray(ClassTUMORpre))
ClassTUMORpre=ClassTUMORpre[~np.isnan(ClassTUMORpre)]
X_1 = np.stack((ClassNAWMpost,ClassNAWMpre)) # we only take the first two features.
X_2 = np.stack((ClassTUMORpost,ClassTUMORpre))
X=np.concatenate((X_1.transpose(), X_2.transpose()),axis=0)
y =np.zeros((np.shape(X))[0])
y[np.shape(X_1)[1]:]=1
print(y)
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# have a look at the descission tree
dot_data = StringIO()
export_graphviz(model, out_file='test.dot',
                         feature_names=['Intensity_Post', 'Intensity_Pre'],
                         class_names=['White Matter', 'Tumor'],
                         filled=True, rounded=True,
                         special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("DT.png")
