import numpy as np
import cv2


NUM_INDIVIDUALS = 40

NUM_SAMPLE = 2


# Returns a matrix containing NUM_INDIVIDUAL rows
# of each picture named "id.pgm"
def load_images(i):
    database = []
    for path in xrange(0, NUM_INDIVIDUALS):
        im = cv2.imread("../faces_full/s%d/%d.pgm" % (path+1, i), flags=cv2.IMREAD_GRAYSCALE)
        database.append(np.ravel(im))
    return np.stack(database)


# Returns a vector with Vi = weight of the i-th
# eigenface relative to the face passed as parameter
def calculate_omega(eigfaces, face, average_face):
    projected = []
    for x in range(0, (eigfaces.shape[0])):
        projected.append(np.dot(eigfaces[x].transpose(), face - average_face))
    projected = np.stack(projected)
    return projected / np.linalg.norm(projected, 2)


def normalize_matrix(m):
    ans = []
    for p in range(0, m.shape[0]):
        ans.append(m[p] / np.linalg.norm(m[p]))
    ans = np.stack(ans)
    return ans


average_omegas = []
outer_eigenfaces = None
outer_mean = None


for i in range(0, NUM_SAMPLE):

    # Loading raw data and preliminary processing
    db = load_images(i+1)
    mean = np.int_(db.mean(0))
    dbc = db - mean

    # PseudoCovariance matrix
    pseudo = np.dot(dbc, dbc.transpose())
    ps_eigenvals, ps_eigenvecs = np.linalg.eig(pseudo)

    # Calculating eigenfaces (returning to R^(h*v)
    eigenfaces = []
    for x in range(0, (ps_eigenvecs.shape[0])):
        eigenfaces.append(np.dot(dbc.transpose(), ps_eigenvecs[x]))
    eigenfaces = np.stack(eigenfaces)

    normalized_eigenfaces = normalize_matrix(eigenfaces)

    # Calculating weighs of eigenfaces relative to the
    # i-th face of the database
    for j in range(0, NUM_INDIVIDUALS):
        if i == 0:
            average_omegas.append(calculate_omega(normalized_eigenfaces, db[j], mean))
        else:
            average_omegas[j] += calculate_omega(normalized_eigenfaces, db[j], mean)

    if i == 0:
        outer_mean = mean
        outer_eigenfaces = eigenfaces
    else:
        outer_mean += mean
        outer_eigenfaces += eigenfaces

outer_mean /= NUM_SAMPLE
outer_eigenfaces /= NUM_SAMPLE
normalized_outer_eigenfaces = normalize_matrix(outer_eigenfaces)

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

counter = 0

for j in range(0, NUM_INDIVIDUALS):
    min_distance = None
    matching = None
    test_omega = calculate_omega(normalized_outer_eigenfaces, new_db[j], outer_mean)
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



