from lib import *


class Sample:

    def __init__(self, eigenfaces, mean, omegas):
        self.eigenfaces = eigenfaces
        self.mean = mean
        self.omegas = omegas

    def compare(self, other):
        ans = []
        for i in range(0, NUM_INDIVIDUALS):
            ans.append(np.linalg.norm(self.omegas[i] - other))
        return ans


