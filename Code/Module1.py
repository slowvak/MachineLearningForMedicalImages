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

