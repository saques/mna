from svm import universe
from sklearn import svm
from kpca import KPCA
from pca import PCA
from lib import *
from parameters import *


PICTURE = 10
imgperper = 5


def kpca_universe(PICTURE, imgperper, NUM_INDIVIDUALS):
    kpca = KPCA()
    db, classes = kpca.get_default_db(imgperper)
    kpca.set_db(db, classes)

    clf = svm.LinearSVC()  # Already implemented SVM
    eigenvectors, eig_classes = kpca.calculate_eigenfaces()
    clf.fit(eigenvectors, eig_classes)

    db_test = load_images(PICTURE, NUM_INDIVIDUALS)
    counter = 0
    class_predicted = clf.predict(kpca.project_images(db_test))

    for i in range(0, NUM_INDIVIDUALS):
        if class_predicted[i] == i:
            counter +=1
    return counter / float(NUM_INDIVIDUALS) * 100


for j in range(1, imgperper, 2):
    Parameters.trnperper = j
    for i in range(j+1, PICTURE, 1):
        print "PICTURE:%d; imgperper:%d" % (i, j)
        print kpca_universe(i, j, NUM_INDIVIDUALS)
