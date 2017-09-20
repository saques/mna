from svm import universe


PICTURE = 10
imgperper = 10
NUM_INDIVIDUALS = 5



for j in range(1, imgperper, 2):
    for i in range(j, PICTURE, 2):
        print "PICTURE:%d; imgperper:%d" % (i, j)
        print universe(i, j, NUM_INDIVIDUALS)

