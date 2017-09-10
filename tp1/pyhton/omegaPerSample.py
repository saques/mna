from Sample import *
import numpy as np


NUM_SAMPLE = 5

samples = {}

for i in range(0, NUM_SAMPLE):

    # Loading raw data and preliminary processing
    db = load_images(i+1)
    mean = np.int_(db.mean(0))
    dbc = db - mean

    # PseudoCovariance matrix
    pseudo = np.dot(dbc, dbc.transpose())
    ps_eigenvals, ps_eigenvecs = np.linalg.eig(pseudo)

    eigenfaces = []
    omegas = []

    # Calculating eigenfaces (returning to R^(h*v)
    for x in range(0, (ps_eigenvecs.shape[0])):
        eigenfaces.append(np.dot(dbc.transpose(), ps_eigenvecs[x]))

    eigenfaces = np.stack(eigenfaces)

    # Calculating weighs of eigenfaces relative to the
    # i-th face of the database
    for j in range(0, NUM_INDIVIDUALS):
        omegas.append(calculate_omega(eigenfaces, dbc[j]))

    omegas = np.stack(omegas)
    omegas = normalize_matrix(omegas)

    samples[i] = Sample(eigenfaces, mean, omegas)


# Testing against other set of faces
other_db = load_images(10)
other_mean = np.int_(other_db.mean(0))
other_dbc = other_db - other_mean

counter = 0

for i in range(0, NUM_INDIVIDUALS):

    addition = None

    for j in range(0, NUM_SAMPLE):

        sample = samples[j]

        other_omega = calculate_omega(sample.eigenfaces, other_dbc[i])

        res = sample.compare(other_omega)

        if addition is None:
            addition = res
        else:
            addition += res

    min_val = addition[0]
    position = 0

    for x in range(1, NUM_INDIVIDUALS):
        if addition[x] < min_val:
            position = x
            min_val = addition[x]

    if position == i:
        counter += 1


# The match percentage
print ((float(counter)/NUM_INDIVIDUALS)*100)




