from svm import universe
from sklearn import svm
from kpca import KPCA
from lib import *
import numpy as np


PICTURE = 10
imgperper = 10
NUM_INDIVIDUALS = 5


def kpca_universe(PICTURE, imgperper, NUM_INDIVIDUALS):
    kpca = KPCA()
    db, classes = kpca.get_default_db(imgperper, NUM_INDIVIDUALS)
    kpca.set_db(db)

    clf = svm.LinearSVC()  # Already implemented SVM
    clf.fit(kpca.calculate_eigenfaces(), classes)

    db_test = load_images(PICTURE, NUM_INDIVIDUALS)
    counter = 0
    class_predicted = clf.predict(kpca.project_images(db_test))

    for i in range(0, NUM_INDIVIDUALS):
        if class_predicted[i] == i:
            counter +=1
    return counter / float(NUM_INDIVIDUALS) * 100


for j in range(1, imgperper, 2):
    for i in range(j+1, PICTURE, 1):
        print "PICTURE:%d; imgperper:%d" % (i, j)
        print kpca_universe(i, j, NUM_INDIVIDUALS)
