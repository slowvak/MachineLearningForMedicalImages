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
h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.1, C=10).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Intensity post contrast')
    plt.ylabel('Intensity pre contrast')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()


# understanding margins 

for C in [1,2,3,10]:
	fig = plt.subplot()
	clf = svm.SVC(C,kernel='linear')
	clf.fit(X, y)
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx = np.linspace(x_min,x_max)
	# print (xx)
	xx=np.asarray(xx)
	# get the separating hyperplane
	w = clf.coef_[0]
	# print(w)
	a = -w[0] / w[1]
	# print (a)
	yy = a * xx - (clf.intercept_[0]) / w[1]
	# print(yy)
	# plot the parallels to the separating hyperplane that pass through the
	# support vectors
	b = clf.support_vectors_[0]
	yy_down = a * xx + (b[1] - a * b[0])
	b = clf.support_vectors_[-1]
	yy_up = a * xx + (b[1] - a * b[0])

	# plot the line, the points, and the nearest vectors to the plane
	plt.plot(xx, yy, 'k-')
	plt.plot(xx, yy_down, 'k--')
	plt.plot(xx, yy_up, 'k--')

	plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
	            s=80, facecolors='none')
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
	plt.axis('tight')
	plt.show()



# compare all classifiers




h = .02  # step size in the mesh
linearly_separable = (X, y)
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


datasets = [linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()
