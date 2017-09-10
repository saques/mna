from lib import *
import numpy as np


NUM_SAMPLE = 2

average_omegas = []
eigenfaces = []

for i in range(0, NUM_SAMPLE):

    # Loading raw data and preliminary processing
    db = load_images(i+1)
    mean = np.int_(db.mean(0))
    dbc = db - mean

    # PseudoCovariance matrix
    pseudo = np.dot(dbc, dbc.transpose())
    ps_eigenvals, ps_eigenvecs = np.linalg.eig(pseudo)

    # Calculating eigenfaces (returning to R^(h*v)
    for x in range(0, (ps_eigenvecs.shape[0])):
        if i == 0:
            eigenfaces.append(np.dot(dbc.transpose(), ps_eigenvecs[x]))
        else:
            v = 1/(1+i)
            eigenfaces[x] *= (1-v)
            eigenfaces[x] += v*np.dot(dbc.transpose(), ps_eigenvecs[x])

    if i == 0:
        eigenfaces = np.stack(eigenfaces)

    normalized_eigenfaces = normalize_matrix(eigenfaces)

    # Calculating weighs of eigenfaces relative to the
    # i-th face of the database
    for j in range(0, NUM_INDIVIDUALS):
        if i == 0:
            average_omegas.append(calculate_omega(normalized_eigenfaces, dbc[j]))
        else:
            average_omegas[j] += calculate_omega(normalized_eigenfaces, dbc[j])

    if i == 0:
        outer_mean = mean
    #     outer_eigenfaces = eigenfaces
    else:
        outer_mean += mean
    #     outer_eigenfaces += eigenfaces

outer_mean /= NUM_SAMPLE
# outer_eigenfaces /= NUM_SAMPLE
normalized_outer_eigenfaces = normalize_matrix(eigenfaces)

average_omegas = np.stack(average_omegas)
average_omegas /= NUM_SAMPLE

average_omegas = normalize_matrix(average_omegas)

# Up to this point we have built a matrix containing in each
# the mean omega for each face in the database, taking into
# account the first NUM_SAMPLE faces. Therefore, we can now
# compare any of the possible faces against this values and
# compute the euclidean distance between a given face's omega
# value and each omega in this set. The smallest euclidean
# distance should tell which face we are talking about.


# Testing against a set of faces not present in the omega's set
new_db = load_images(7)
new_mean = np.int_(db.mean(0))
new_dbc = new_db - new_mean

counter = 0

for j in range(0, NUM_INDIVIDUALS):
    min_distance = None
    matching = None
    test_omega = calculate_omega(normalized_outer_eigenfaces, new_dbc[j])
    for i in range(0, NUM_INDIVIDUALS):
        distance = np.linalg.norm(average_omegas[i]-test_omega)
        if i == 0:
            min_distance = distance
            matching = i
        elif distance < min_distance:
            min_distance = distance
            matching = i

    if matching == j:
        counter += 1


# The match percentage
print ((float(counter)/NUM_INDIVIDUALS)*100)



