# from lib import *
import numpy as np
from sklearn import svm

from pyhton.lib import load_images_3, normalize_matrix, calculate_omega, load_images

omegas = []
eigenfaces = []

PICTURE = 6
imgperper = 4
NUM_INDIVIDUALS = 30

# AVERAGE

# 24 <- 6/4/30
# 23 <- 10/4/30
# 28 <- 2/4/30

# 26 <- 10/6/30
# 23 <- 8/6/30
# 24 <- 7/6/30
# 30 <- 3/6/30

# 20 <- 3/6/20
# 16 <- 9/6/20
# 17 <- 8/6/20

# WITHOUT AVERAGE

# 24 <- 6/4/30
# 22 <- 10/4/30
# 27 <- 2/4/30

# 23 <- 10/6/30
# 20 <- 8/6/30
# 24 <- 7/6/30
# 26 <- 3/6/30

# 19 <- 3/6/20
# 14 <- 9/6/20
# 16 <- 8/6/20



# Loading raw data and preliminary processing
db = load_images_3(imgperper, NUM_INDIVIDUALS)


mean = np.int_(db.mean(0))
dbc = db - mean

# PseudoCovariance matrix
pseudo = np.dot(dbc, dbc.transpose())
ps_eigenvals, ps_eigenvecs = np.linalg.eig(pseudo)

ps_eigenvals = np.absolute(ps_eigenvals)

ps_indexorder = np.argsort(ps_eigenvals);

ps_sortedevectors = []

for x in ps_indexorder:
    ps_sortedevectors.append(ps_eigenvecs[x])

# Calculating eigenfaces (returning to R^(h*v)
for x in range(0, ps_sortedevectors.shape[0]):
    eigenfaces.append(np.dot(dbc.transpose(), ps_sortedevectors[x]))
eigenfaces = np.stack(eigenfaces)

eigenfaces = normalize_matrix(eigenfaces)

eigenfaces_alt = []

for i in xrange(0, NUM_INDIVIDUALS):
    eigenfaces_alt.append(eigenfaces[i])

eigenfaces = np.stack(eigenfaces_alt)


# Calculating weighs of eigenfaces relative to the
# i-th face of the database

# AVERAGE

for j in range(0, NUM_INDIVIDUALS):
    avg = None
    for i in range(0, imgperper):
        if i == 0:
            avg = calculate_omega(eigenfaces, dbc[j*imgperper+i])
        else:
            avg += calculate_omega(eigenfaces, dbc[j*imgperper+i])

    avg = np.divide(avg, imgperper)

    omegas.append(avg)

omegas = np.stack(omegas)

# WITHOUT AVERAGE

# for i in range(0, NUM_INDIVIDUALS*imgperper):
#     omegas.append(calculate_omega(eigenfaces, dbc[i]))
# omegas = np.stack(omegas)


omegas = normalize_matrix(omegas)

classes = []

#AVERAGE

for j in xrange(0, NUM_INDIVIDUALS):
    classes.append(j)


# WITHOUT AVERAGE

# for i in range(0, NUM_INDIVIDUALS):
#     for j in range(0, imgperper):
#         classes.append(i)

classes = np.stack(classes)


#Building test set
db_test = load_images(PICTURE)
db_test = db_test - mean
omegas_test = []
for j in range(0, NUM_INDIVIDUALS):
    omegas_test.append(calculate_omega(eigenfaces, db_test[j]))
omegas_test = np.stack(omegas_test)

omegas_test = normalize_matrix(omegas_test)


#Training
clf = svm.SVC()
clf.set_params(probability=True)

clf.fit(omegas, classes)

#individuals 0-39

counter = 0

for i in xrange(0, NUM_INDIVIDUALS):

    if clf.predict(omegas_test[i].reshape(1, -1))[0] == i:
        counter += 1

print counter




