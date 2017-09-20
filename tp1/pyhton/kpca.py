from lib import *
import numpy as np
import cv2
from sklearn import svm

'''
Facial Recognition using KPCA
'''

# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92
areasize = IMG_HEIGHT * IMG_WIDTH

personsno = NUM_INDIVIDUALS
tstperper = 8
trnperper = 10
tstno = tstperper * personsno
trnno = trnperper * personsno

class KPCA:

    # Class Variables
    degree = 2

    def __init__(self):
        # Instance Variables
        self.eigenvectors = []
        self.K = None
        self.db = None
        self.f = 1

    def image_proyection_from_path(self, img_path):
        # Load training and images and their classes
        db, classes = load_images_and_get_class(trnperper)
        tstimg = np.ravel(cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE))

        # Make every pixel have a value between -1 and 1
        db = np.divide(np.add(db, -127.5), 127.5)
        tstimg = np.divide(np.add(tstimg, -127.5), 127.5)

        print self.project_normalized_images(tstimg)

    def project_images(self, imgs):
        return self.project_normalized_images(np.divide(np.add(imgs, -127.5), 127.5))

    def project_normalized_images(self, imgs):
        # Project test images
        onesM_tst = np.ones([tstno, trnno]) / trnno
        oneM = np.ones([trnno, trnno]) / trnno
        Ktest = (np.dot(imgs, self.db.T) / trnno + 1) ** self.degree
        Ktest = Ktest - np.dot(onesM_tst, self.K) - np.dot(Ktest, oneM) + np.dot(onesM_tst, np.dot(self.K, oneM))
        return np.dot(Ktest, self.eigenvectors)

    def calculate_eigenfaces(self):
        if self.db is None:
           print "No database set"
           return None

        # Polynomial kernel
        K = (np.dot(self.db, self.db.T) / trnno + 1) ** self.degree
        oneM = np.ones([trnno, trnno]) / trnno
        K = K - np.dot(oneM, K) - np.dot(K, oneM) + np.dot(oneM, np.dot(K, oneM))
        self.K = K

        # Get eigenvectors and sort them by their associated eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(K)   # sorted in ascending order
        eigenvalues = np.flipud(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)

        # Normalize eigenvectors
        for col in range(eigenvectors.shape[1]):
            eigenvectors[:, col] = eigenvectors[:, col] / np.sqrt(np.abs(eigenvalues[col]))
        self.eigenvectors = eigenvectors

        # Project training images
        return np.dot(K.T, eigenvectors)

    def get_default_db(self, imperper):
        return load_images_and_get_class(imperper)

    def set_db(self, db):
        self.db = np.divide(np.add(db, -127.5), 127.5)


    # def main():
    #     # Load training and images and their classes
    #     db, classes = load_images_and_get_class(trnperper)
    #     tstdb, tstclasses = load_images_and_get_class(tstperper)
    #
    #     # Make every pixel have a value between -1 and 1
    #     db = np.divide(np.add(db, -127.5), 127.5)
    #     tstdb = np.divide(np.add(tstdb, -127.5), 127.5)
    #
    #
    #     kpca = KPCA()
    #
    #     # Fit the SVM
    #     clf = svm.LinearSVC()   #Already implemented SVM
    #     clf.fit(kpca.calculate_eigenfaces(db),classes.ravel())
    #     # print clf.score(kpca.project_images(tstdb),tstclasses.ravel()) * 100
    #     print clf.predict(kpca.project_images(tstdb))
    #     kpca.image_proyection_from_image("../orl_faces/s4/2.pgm")


    # if __name__ == "__main__":
    #     main()



    # # Cheating with already implemented KPCA
    # kpca = KernelPCA(n_components = None, kernel='poly', degree=2)
    # kpca.fit(db)
    #
    # improypre2 = kpca.fit_transform(db)
    # imtstproypre2 = kpca.transform(tstdb)
    #
    # # Already implemented SVM
    # clf = svm.LinearSVC()   # sklearn KCPA implementation
    # clf.fit(improypre2,classes.ravel())
    # print clf.score(imtstproypre2,tstclasses.ravel()) * 100


    # nmax = db.shape[1]
    # accs = np.zeros([nmax, 1])
    # for neigen in range(1, nmax):
    #     # Me quedo solo con las primeras autocaras
    #     # proyecto
    #     improy = improypre[:, 0:neigen]
    #     imtstproy = imtstproypre[:, 0:neigen]
    #
    #     # SVM
    #     # entreno
    #     clf = svm.LinearSVC()
    #     clf.fit(improy,classes.ravel())
    #     accs[neigen] = clf.score(imtstproy,tstclasses.ravel())
    #     print('Precision con {0} autocaras: {1} %\n'.format(neigen, accs[neigen] * 100))
