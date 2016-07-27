#  Demonstration code developed bu Mayo Clinic Radiology Informatics Laboratory
# Copyright 2016
#
import os
#
# Module 1 - Data Load / Display / Normalization
#
# we will directly read DICOM files. We will use a library called pyDicom
# if you get an error on this statement, you need to install the library:
# pip install pydicom
import dicom as pydicom
# same for these libraries
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from matplotlib import cm

CurrentDir, CurrentFile = os.path.split(__file__)


# Here are the file names--these files should be in the same directory as this file
PreName =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, "Data",'Pre.dcm') )  
PostName =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, "Data",  'Post.dcm')  )
T2Name =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, "Data", 'T2.dcm') )  
FLAIRName =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, "Data", 'FLAIR.dcm') )  

# read Pre in--we assume that all images are same x,y dims
Pre = pydicom.read_file(PreName)
xdim = (int) (Pre.Rows)
ydim = (int) (Pre.Columns) 

# make space in a numpy array for the images
ArrayDicom = np.zeros((xdim, ydim, 4), dtype=Pre.pixel_array.dtype)

# copy Pre pixels into z=0
ArrayDicom[:, :, 0] = Pre.pixel_array  
# Post
Post = pydicom.read_file(PostName)
ArrayDicom[:, :, 1] = Post.pixel_array  
# T2
T2 = pydicom.read_file(T2Name)
ArrayDicom[:, :, 2] = T2.pixel_array  
#FLAIR
FLAIR = pydicom.read_file(FLAIRName)
ArrayDicom[:, :, 3] = FLAIR.pixel_array  
print ("Data Loaded")


# note that the 4 brain slices are all aligned
# the rectangles here are known to represent the tissues of interest:
# Tissue0 = NAWM, Tissue 1 = GM, Tissue2 = CSF, Tissue3 = Air
NAWM = 0
GM = 1
CSF = 2
AIR = 3
TUMOR = 4
# each tissue has x1, y1, x2, y2 as rectangle in image that represents that tissue
Tissues = np.zeros((5, 2, 2), dtype=np.int32)
Tissues[NAWM][0][0] = 145 # we measured the location of these tissues
Tissues[NAWM][0][1] = 68
Tissues[NAWM][1][0] = 157
Tissues[NAWM][1][1] = 74

Tissues[GM][0][0] = 166 # we measured the location of these tissues
Tissues[GM][0][1] = 58
Tissues[GM][1][0] = 175
Tissues[GM][1][1] = 62

Tissues[CSF][0][0] = 105 # we measured the location of these tissues
Tissues[CSF][0][1] = 148
Tissues[CSF][1][0] = 120
Tissues[CSF][1][1] = 154

Tissues[AIR][0][0] = 1 # we measured the location of these tissues
Tissues[AIR][0][1] = 1
Tissues[AIR][1][0] = 10
Tissues[AIR][1][1] = 10

Tissues[TUMOR][0][0] = 65 # we measured the location of these tissues
Tissues[TUMOR][0][1] = 130
Tissues[TUMOR][1][0] = 74
Tissues[TUMOR][1][1] = 139

plt.figure()
plt.imshow(np.flipud(Pre.pixel_array),cmap='gray')
plt.contour(Tissues[NAWM], colors='k', origin='image')
plt.axis([0,256,0,256])
plt.show()


# Create classes 
ClassNAWMpost=ArrayDicom[Tissues[NAWM][0][0]:Tissues[NAWM][1][0],Tissues[NAWM][0][1]:Tissues[NAWM][1][1],0]
ClassNAWMpret=ArrayDicom[Tissues[NAWM][0][0]:Tissues[NAWM][1][0],Tissues[NAWM][0][1]:Tissues[NAWM][1][1],1]
ClassNAWMT2=ArrayDicom[Tissues[NAWM][0][0]:Tissues[NAWM][1][0],Tissues[NAWM][0][1]:Tissues[NAWM][1][1],2]
ClassNAWMFLAIR=ArrayDicom[Tissues[NAWM][0][0]:Tissues[NAWM][1][0],Tissues[NAWM][0][1]:Tissues[NAWM][1][1],3]
ClassTUMORpost=ArrayDicom[Tissues[TUMOR][0][0]:Tissues[TUMOR][1][0],Tissues[TUMOR][0][1]:Tissues[TUMOR][1][1],0]
ClassTUMORpret=ArrayDicom[Tissues[TUMOR][0][0]:Tissues[TUMOR][1][0],Tissues[TUMOR][0][1]:Tissues[TUMOR][1][1],1]
ClassTUMORT2=ArrayDicom[Tissues[TUMOR][0][0]:Tissues[TUMOR][1][0],Tissues[TUMOR][0][1]:Tissues[TUMOR][1][1],2]
ClassTUMORFLAIR=ArrayDicom[Tissues[TUMOR][0][0]:Tissues[TUMOR][1][0],Tissues[TUMOR][0][1]:Tissues[TUMOR][1][1],3]


# Display Tumor vs NAWM
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ClassNAWMpost.reshape(-1), ClassNAWMpret.reshape(-1), ClassNAWMFLAIR.reshape(-1))
ax.scatter(ClassTUMORpost.reshape(-1), ClassTUMORpret.reshape(-1), ClassTUMORFLAIR.reshape(-1), c='r', marker='^')
ax.set_xlabel('post')
ax.set_ylabel('pret')
ax.set_zlabel('FLAIR')
plt.show()

# run svm classifiers

X_1 = np.stack((ClassNAWMpost.reshape(-1),ClassNAWMpret.reshape(-1)), axis=-1) # we only take the first two features.
X_2 = np.stack((ClassTUMORpost.reshape(-1),ClassTUMORpret.reshape(-1)),axis=-1)  
# print (X_1)
# print (np.shape(X_1))
# print (X_2)
# print (np.shape(X_2))  
X=np.vstack((X_1,X_2))
y =np.zeros((np.shape(X))[0])
y[np.shape(X_1)[0]:]=1


# load the iris datasets
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