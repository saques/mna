from lib import *
import numpy as np
import cv2
from sklearn.decomposition import KernelPCA
from sklearn import svm

'''
Facial Recognition using KPCA
'''

# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92
areasize = IMG_HEIGHT * IMG_WIDTH

personsno = NUM_INDIVIDUALS
tstperper = 7
trnperper = 6
tstno = tstperper * personsno
trnno = trnperper * personsno

# Load training and test images and their classes
db, classes = load_images_and_get_class(trnperper)
tstdb, tstclasses = load_images_and_get_class(tstperper)

# Make every pixel have a value between -1 and 1
db = np.divide(np.add(db,-127.5),127.5)
tstdb = np.divide(np.add(tstdb,-127.5),127.5)

# Polynomial kernel
degree = 5
K = (np.dot(db, db.T) / trnno + 1) ** degree
oneM = np.ones([trnno, trnno]) / trnno
K = K - np.dot(oneM, K) - np.dot(K, oneM) + np.dot(oneM, np.dot(K, oneM))

# Get eigenvectors and sort them by their associated eigenvalue
eigenvalues, eigenvectors = np.linalg.eigh(K)   # sorted in ascending order
eigenvalues = np.flipud(eigenvalues)
eigenvectors = np.fliplr(eigenvectors)

# Normalize eigenvectors
for col in range(eigenvectors.shape[1]):
    eigenvectors[:,col] = eigenvectors[:,col]/np.sqrt(np.abs(eigenvalues[col]))

# Project training images
improypre = np.dot(K.T, eigenvectors)

# Project test images
onesM_tst = np.ones([tstno, trnno]) / trnno
Ktest = (np.dot(tstdb, db.T) / trnno + 1) ** degree
Ktest = Ktest - np.dot(onesM_tst, K) - np.dot(Ktest, oneM) + np.dot(onesM_tst, np.dot(K, oneM))
imtstproypre = np.dot(Ktest, eigenvectors)


# Already implemented SVM
clf = svm.LinearSVC()   # our KPCA implementation
clf.fit(improypre,classes.ravel())
print clf.score(imtstproypre,tstclasses.ravel()) * 100



# Cheating with already implemented KPCA
kpca = KernelPCA(n_components=None, kernel='poly', degree=1, gamma = 1, coef0 = 0)
kpca = KernelPCA(n_components = None, kernel='poly', degree=2)
kpca.fit(db)

improypre2 = kpca.fit_transform(db)
imtstproypre2 = kpca.transform(tstdb)

# Already implemented SVM
clf = svm.LinearSVC()   # sklearn KCPA implementation
clf.fit(improypre2,classes.ravel())
print clf.score(imtstproypre2,tstclasses.ravel()) * 100
